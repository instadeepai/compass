# Copyright 2023 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import TYPE_CHECKING, Any, Optional, Tuple

import chex
from chex import Array

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

import numpy as np
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import rlax
import omegaconf
from jumanji.types import TimeStep

from compass.environments.poppy_env import PoppyEnv
from compass.networks import Networks
from compass.trainers.trainer_utils import ActingState, Information, Observation, State
from compass.trainers.validation_utils import get_params, load_instances
from compass.utils.data import prepare_problem_batch
from compass.utils.utils import spread_over_devices


@dataclass
class ActingStateDeterministic:  # type: ignore
    """Container for data used during the acting in the environment: no key as it's for deterministic action choice."""

    state: State
    timestep: TimeStep


def get_optimizer(cfg: omegaconf.DictConfig) -> optax.GradientTransformation:
    optimizer = optax.adam(cfg.learning_rate)
    optimizer = optax.MultiSteps(optimizer, cfg.num_gradient_accumulation_steps)
    return optimizer


def get_networks(cfg) -> Networks:
    def encoder_fn(problem: chex.Array):
        encoder = hydra.utils.instantiate(cfg.encoder, name="shared_encoder")
        return encoder(problem)

    def decoder_fn(observation: Observation, embeddings: Array):
        decoder = hydra.utils.instantiate(cfg.decoder, name="decoder")
        return decoder(observation, embeddings)

    return Networks(
        encoder_fn=hk.without_apply_rng(hk.transform(encoder_fn)),
        decoder_fn=hk.without_apply_rng(hk.transform(decoder_fn)),
    )


def get_instances(cfg, key, environment, params, num_start_positions, pmap_over='problems'):
    """Get the problem instances, start positions, and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: A PRNGKey.
        environment: The environment to generate the starting positions on.
        params: The encoder and decoder parameters.
        num_start_positions: The number of starting positions to generate.
        pmap_over: The name of the variable to pmap over (either 'problems' or 'augmentations').

    Returns:
        problems: A batch of N problems divided over D devices ([D, N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair divided over D devices
        ([D, N, K, M]).
        acting_keys: M acting keys for each problem-agent pair divided over D devices
        ([D, N, K, M, 2]).
    """
    _, decoder = hk.data_structures.partition(lambda m, n, p: "encoder" in m, params)
    num_agents = jax.tree_util.tree_leaves(decoder)[0].shape[0]

    if cfg.load_problem:
        problems, start_positions, acting_keys = load_instances(
            cfg, key, environment, num_start_positions, num_agents
        )
    else:
        problems, start_positions, acting_keys = prepare_problem_batch(
            key, environment, cfg.num_problems, num_agents, num_start_positions
        )

    print("Number of problems:", problems.shape[0])

    if pmap_over == 'augmentations':
        get_augmentations = environment.get_augmentations if cfg.use_augmentations else lambda x: x[None]
        problems = jax.vmap(get_augmentations)(problems)  # [N, augs, problem_size, 2]
        problems = problems.transpose((1, 0, 2, 3))  # [augs, N, problem_size, 2]

        # repeat start positions and acting keys for each augmentation
        start_positions = jnp.repeat(start_positions, problems.shape[0], axis=0)

    problems = spread_over_devices(problems)
    start_positions = spread_over_devices(start_positions)

    if pmap_over == 'augmentations':
        problems = problems.squeeze(1)

    return problems, start_positions, acting_keys


def generate_trajectory(
        decoder_apply_fn,
        policy_temperature,
        environment,
        problem,
        embeddings,
        params,
        start_position,
        acting_key,
):
    """Decode a single agent, from a single starting position on a single problem.

    With decorators, the expected input dimensions are:
        problems: [N, problem_size, 2]
        embeddings: [N, problem_size, 128]
        params (decoder only): {key: [K, ...]}
        start_position: [N, K, M]
        acting_key: [N, K, M, 2]
    """

    def policy(
            observation: Observation,
            key,
    ) -> Array:
        logits = decoder_apply_fn(params, observation, embeddings)
        logits -= 1e30 * observation.action_mask
        if policy_temperature > 0:
            action = rlax.softmax(temperature=policy_temperature).sample(key, logits)
        else:
            action = rlax.greedy().sample(key, logits)
        logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
        return action, logprob

    def take_step(acting_state):
        # TODO when the environment is done, a dummy step should be used to save computation time.
        #  Especially useful for knapsack environment where real number of steps << max number of steps
        #  theoretically possible.
        key, act_key = random.split(acting_state.key, 2)
        action, logprob = policy(acting_state.timestep.observation, act_key)
        state, timestep = environment.step(acting_state.state, action)
        info = Information(extras={"logprob": logprob, "action": action}, metrics={}, logging={})
        acting_state = ActingState(state=state, timestep=timestep, key=key)
        return acting_state, (timestep, info)

    state, timestep = environment.reset_from_state(problem, start_position)

    acting_state = ActingState(state=state, timestep=timestep, key=acting_key)

    acting_state, (traj, info) = jax.lax.scan(
        lambda acting_state, _: take_step(acting_state),
        acting_state,
        xs=None,
        length=environment.get_episode_horizon(),
    )

    return acting_state, (traj, info)


def calculate_loss(traj, info, use_poppy=False, augmentations=False) -> chex.Array:
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # Calculate advantages.
    if returns.shape[-1] > 1:
        baseline = returns.mean(axis=(-1), keepdims=True)
        if augmentations:
            baseline = jax.lax.pmean(baseline, axis_name='i')
        advantages = returns - baseline
    else:
        advantages = returns

    if use_poppy:
        train_idxs = returns.argmax(axis=1, keepdims=True)
        advantages = jnp.take_along_axis(advantages, train_idxs, axis=1)
        logprob_traj = jnp.take_along_axis(logprob_traj, train_idxs, axis=1)

    loss = -jnp.mean(advantages * logprob_traj)
    return loss


def eas_rollout(
        cfg: omegaconf.DictConfig,
        environment: PoppyEnv,
        params: chex.ArrayTree,
        networks: Networks,
        problems: jnp.ndarray,
        start_positions: jnp.ndarray,
        keys: jnp.ndarray,
) -> Tuple[ActingState, Tuple[TimeStep, Information]]:
    """Rollout a batch of agents on a batch of problems and starting points.
    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
            across all agents. There is only one decoder in the case of conditioned decoder. A population
            is implicitely created by the use of several behavior markers as input to the decoder.
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    Returns:
        # TODO
    """

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0))  # over N problems
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))  # over K agents
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0))  # M starting pos.
    def rollout(problem, embeddings, params, start_position, acting_key):
        """Decode a single agent, from a single starting position on a single problem.

        With decorators, the expected input dimensions are:
            problems: [N, problem_size, 2]
            embeddings: [N, problem_size, 128]
            params (decoder only): {key: [K, ...]}
            start_position: [N, K, M]
            acting_key: [N, K, M, 2]
        """

        return generate_trajectory(networks.decoder_fn.apply, cfg.policy.temperature, environment, problem,
                                   embeddings, params, start_position, acting_key)

    def generate_specific_trajectory(problem, embeddings, params, start_position, action_sequence):
        """Decode a single agent, from a single starting position on a single problem, with a specific action sequence.
        """

        def policy_predefined(
                observation: Observation,
                action: Array,
        ) -> Array:
            logits = networks.decoder_fn.apply(params, observation, embeddings)
            logits -= 1e30 * observation.action_mask
            logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
            return action, logprob

        def take_step_predefined(acting_state: ActingStateDeterministic,
                                 action: Array, ) -> Tuple[ActingStateDeterministic, Information]:
            action, logprob = policy_predefined(acting_state.timestep.observation, action)
            state, timestep = environment.step(acting_state.state, action)
            info = Information(extras={"logprob": logprob}, metrics={}, logging={})
            acting_state = ActingStateDeterministic(state=state, timestep=timestep)
            return acting_state, (timestep, info)

        state, timestep = environment.reset_from_state(problem, start_position)

        acting_state = ActingStateDeterministic(state=state, timestep=timestep)

        acting_state, (traj, info) = jax.lax.scan(
            lambda acting_state, action: take_step_predefined(acting_state, action),
            acting_state,
            xs=action_sequence,
            length=environment.get_episode_horizon(),
        )

        return acting_state, (traj, info)

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0))  # over N problems
    @functools.partial(jax.vmap, in_axes=(None, None, 0, None, None))  # over K agents (even if only one)
    def generate_winning_trajectories(problem, embeddings, params, start_position, action_sequences):
        return generate_specific_trajectory(problem, embeddings, params, start_position, action_sequences)

    # split the params in encoder and decoder - those a merged in the training state
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # initialise the embeddings for each problem
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )
    optimizer_state = get_optimizer(cfg.optimizer).init(
        embeddings
    )
    def eas_loss(traj, info, best_info, imitation_coef=1) -> chex.Array:
        rl_loss = calculate_loss(traj, info, use_poppy=False, augmentations=cfg.use_augmentations)
        if imitation_coef == 0:
            print('skipping imitation loss computation')
            return rl_loss
        print("Includes imitation loss computation")
        imitation_loss = - best_info.extras["logprob"].sum(-1).mean()
        return rl_loss + imitation_coef * imitation_loss

    def loss_and_output(embeddings, decoder_params, start_positions, acting_keys, best_rewards, best_actions,
                        best_start):
        # Sample trajectories using rollout
        acting_state, (traj, info) = rollout(problems, embeddings, decoder_params, start_positions, acting_keys)

        # Keep the best trajectory for each problem
        returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]

        # Calculate new best rewards and log probabilities using the previous solution
        max_indices = jnp.argmax(returns.reshape(returns.shape[0], -1), axis=1)  # [N, K * M] --> [N]
        flat_returns = returns.reshape(returns.shape[0], -1)  # Flatten the K and M dimensions
        new_best_rewards = jnp.take_along_axis(flat_returns, max_indices[:, None], axis=1)[:, 0]  # [N]

        # Unravel the indices to get indices for augs, K, and M dimensions
        k_indices, m_indices = jnp.unravel_index(max_indices, shape=(
            returns.shape[1], returns.shape[2]))  # [N], [N]
        n_indices = jnp.arange(returns.shape[0])
        new_best_actions = info.extras["action"][n_indices, k_indices, m_indices]  # [N, t]

        if cfg.use_augmentations:
            new_best_rewards_augs = jax.lax.all_gather(new_best_rewards, axis_name="i")  # [aug, N]
            new_best_actions_augs = jax.lax.all_gather(new_best_actions, axis_name="i")  # [aug, N, t]
            m_indices_augs = jax.lax.all_gather(m_indices, axis_name="i")  # [aug, N]

            # Step 1: Identify the Best Aug for each N
            best_aug_indices = jnp.argmax(new_best_rewards_augs, axis=0)

            # Step 2: Use the Indices to Gather Data
            new_best_rewards = jnp.take_along_axis(new_best_rewards_augs, best_aug_indices[None, :], axis=0).squeeze()
            new_best_actions = jnp.take_along_axis(new_best_actions_augs, best_aug_indices[None, :, None],
                                                   axis=0).squeeze()
            m_indices = jnp.take_along_axis(m_indices_augs, best_aug_indices[None, :], axis=0).squeeze()

            print("new_best_rewards", new_best_rewards.shape)
            print("new_best_actions", new_best_actions.shape)
            print("m_indices", m_indices.shape)

        update_mask = new_best_rewards > best_rewards
        best_rewards = jnp.where(update_mask, new_best_rewards, best_rewards)  # [N]
        best_start = jnp.where(update_mask, m_indices, best_start)  # [N]
        best_actions = jnp.where(update_mask[:, None], new_best_actions, best_actions)  # [N, t]

        if cfg.imitation_coef > 0:
            print("Calculating best info")
            _, (_, best_info) = generate_winning_trajectories(problems, embeddings, decoder_params, best_start,
                                                              best_actions)
        else:
            best_info = None

        # Calculate the loss using eas_loss
        loss = eas_loss(traj, info, best_info, imitation_coef=cfg.imitation_coef)
        return loss, (traj, info, best_rewards, best_actions, best_start)

    def loop_body(state, _):
        embeddings, decoder_params, start_positions, key, best_rewards, \
            best_actions, best_start, optimizer_state = state

        num_problems, num_agents, num_start_positions = start_positions.shape

        key, act_key = random.split(key)
        acting_keys = random.split(
            act_key, num_problems * num_agents * num_start_positions
        ).reshape((num_problems, num_agents, num_start_positions, -1))

        grads, (traj, info, best_rewards, best_actions, best_start) = jax.grad(loss_and_output, has_aux=True, )(
            embeddings, decoder_params, start_positions, acting_keys, best_rewards, best_actions, best_start
        )

        # no need to average the gradients over the pmap dimension as it's problem dependent
        updates, optimizer_state = get_optimizer(cfg.optimizer).update(
            grads, optimizer_state, params=embeddings
        )
        embeddings = optax.apply_updates(embeddings, updates)

        return (embeddings, decoder_params, start_positions, key, best_rewards,
                best_actions, best_start, optimizer_state), traj.reward.sum(-1).max(-1).max(-1)

    # Run the loop for a fixed number of steps

    # dummy initialisation of the loop variables
    num_problems = embeddings.shape[0]
    best_rewards = -10 ** 3 * jnp.ones(num_problems)
    best_actions = jnp.zeros((num_problems, environment.get_episode_horizon()), dtype=jnp.int32)
    best_start = jnp.zeros(num_problems, dtype=jnp.int32)

    num_steps = cfg.budget
    state = (embeddings, decoder_params, start_positions, keys, best_rewards,
             best_actions, best_start, optimizer_state)
    state, episode_returns = jax.lax.scan(loop_body, state, xs=None, length=num_steps)

    return episode_returns


def eas_emb(
        cfg: omegaconf.DictConfig,
        params: chex.ArrayTree = None,
        logger: Any = None,
) -> dict:
    """Run validation on input problems.

    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.

    Returns:
        metrics: A dictionary of metrics from the validation.
    """

    def log(metrics, used_budget, logger, key=None):
        metrics["used_budget"] = used_budget
        if logger:
            if key:
                metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}
            logger.write(metrics)

    @functools.partial(jax.pmap, axis_name="i")
    def run_eas(problems, start_positions, keys):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).

        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """

        # 1. Split problems, start_positions and acting_keys into chunks of size batch_size.
        # 2. Zip batches into list of inputs:
        #   [(problems[0],start_positions[0],acting_keys[0]),
        #    (problems[1],start_positions[1],acting_keys[1]),
        #    ...]
        num_batches = int(round(len(problems) / cfg.batch_size, 0))
        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        start_positions = jnp.stack(jnp.split(start_positions, num_batches, axis=0))
        keys = jnp.squeeze(keys)
        keys = random.split(keys, num_batches)
        num_problems = problems.shape[1]

        def body(_, x):
            problems, start_positions, keys = x
            returns = eas_rollout(
                cfg=cfg,
                environment=environment,
                params=params,
                networks=networks,
                problems=problems,
                start_positions=start_positions,
                keys=keys,
            )

            return None, returns

        _, episode_return = jax.lax.scan(
            body, init=None, xs=(problems, start_positions, keys)
        )

        # returns = [num_batches, budget, N]

        def flatten_metrics(x):
            x = jnp.transpose(x, (0, 2, 1))
            # x = x.sum(-1)
            # x = jnp.max(x, axis=(-1, -2))  # [num_batches, N, budget]
            return x.reshape(*(-1,) + x.shape[2:])  # [num_problems, budget]

        # Flatten batch dimension of metrics.
        episode_return = flatten_metrics(episode_return)
        return episode_return

    # instantiate networks and environments
    networks = get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)
    if not params:
        params = get_params(cfg.checkpointing)

    # get a set of instances
    key = random.PRNGKey(cfg.problem_seed)

    if cfg.use_augmentations:
        pmap_dimension = 'augmentations'
    else:
        pmap_dimension = 'problems'

    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
        pmap_over=pmap_dimension,
    )

    key = random.split(key, cfg.num_devices)
    key = spread_over_devices(key)

    # run the eas
    episode_return = run_eas(problems, start_positions, key)

    if cfg.use_augmentations:
        episode_return = episode_return.max(0)  # [N, budget]
    else:
        episode_return = jnp.concatenate(episode_return, axis=0)

    # calculate the metrics
    metrics = {
        "episode_return": episode_return.mean(0),
        "best_returns": jax.lax.cummax(episode_return, axis=1).mean(0),
    }

    return metrics
