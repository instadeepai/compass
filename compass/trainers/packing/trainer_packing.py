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

from typing import Tuple

import chex
import omegaconf
import functools
import time
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import optax
from chex import Array
from jax import random
from jumanji.types import TimeStep

import compass.trainers.packing.validation_packing as validation
from compass.environments.jobshop.environment import PoppyJobShop
from compass.environments.jobshop.types import Observation as ObservationJobShop
from compass.utils import (
    create_checkpoint_directory,
    save_checkpoint,
    load_checkpoint_packing,
    prepare_problem_batch,
    fetch_from_first_device,
    reduce_from_devices,
)
from compass.networks.jobshop import make_actor_critic_networks_job_shop, JobShopParams, \
    JobShopNetworks
from compass.trainers.trainer_utils import (
    ActingState,
    Information,
    set_policy,
    TrainingStatePacking as TrainingState,
)


def get_optimizer(cfg: omegaconf.DictConfig) -> optax.GradientTransformation:
    """Create desired optimizer"""
    # define optimizer - encoder/decoder
    optimizer = optax.adam(learning_rate=cfg.lr)

    # use wrapper for gradient accumulation
    optimizer = optax.MultiSteps(optimizer, cfg.num_gradient_accumulation_steps)

    return optimizer


def get_networks(cfg: omegaconf.DictConfig, environment: PoppyJobShop) -> JobShopNetworks:
    return make_actor_critic_networks_job_shop(job_shop=environment,
                                               num_layers_machines=cfg.networks.num_layers_machines,
                                               num_layers_operations=cfg.networks.num_layers_operations,
                                               num_layers_joint_machines_jobs=cfg.networks.num_layers_joint_machines_jobs,
                                               transformer_num_heads=cfg.networks.transformer_num_heads,
                                               transformer_key_size=cfg.networks.transformer_key_size,
                                               transformer_mlp_units=cfg.networks.transformer_mlp_units,
                                               actor_decoder_mlp_units=cfg.networks.actor_decoder_mlp_units,
                                               critic_decoder_mlp_units=cfg.networks.critic_decoder_mlp_units,
                                               )


def init_training_state(
        cfg: omegaconf.DictConfig, networks: JobShopNetworks, environment: PoppyJobShop
) -> TrainingState:
    """Instantiate the initial state of the training process."""

    key = random.PRNGKey(cfg.seed)
    network_key, training_key = random.split(key)

    # load checkpoints
    (
        params,
        optimizer_state,
        keys,
        num_steps,
        extras,
    ) = load_checkpoint_packing(cfg)

    # dummy obs used to create the models params
    _dummy_obs = environment.observation_spec().generate_value()

    if params is None:
        critic_encoder_key, critic_decoder_key, actor_encoder_key, actor_decoder_key = random.split(
            network_key, 4)
        _dummy_behavior_marker = jnp.zeros((1, cfg.behavior_dim), float)
        static_emb = jnp.zeros(
            (1, _dummy_obs.problem.shape[1] + _dummy_obs.problem.shape[2] + 1, 1), float)
        _dummy_obs = jax.tree_map(lambda x: x[None], _dummy_obs)

        critic_encoder_params = networks.critic_encoder.init(critic_encoder_key, _dummy_obs)
        actor_encoder_params = networks.actor_encoder.init(actor_encoder_key, _dummy_obs)

        critic_embedding = networks.critic_encoder.apply(critic_encoder_params, _dummy_obs)
        actor_embedding = networks.actor_encoder.apply(actor_encoder_params, _dummy_obs)

        critic_decoder_params = networks.critic_decoder.init(critic_decoder_key, critic_embedding,
                                                             static_emb, _dummy_behavior_marker)
        actor_decoder_params = networks.actor_decoder.init(actor_decoder_key, actor_embedding,
                                                           static_emb, _dummy_behavior_marker,
                                                           _dummy_obs.action_mask)

        params = JobShopParams(
            critic_encoder=critic_encoder_params,
            critic_decoder=critic_decoder_params,
            actor_encoder=actor_encoder_params,
            actor_decoder=actor_decoder_params,

        )

    # define the behavior markers
    behavior_markers = jnp.zeros((cfg.training_sample_size, cfg.behavior_dim))

    # distribute parameters over devices as required.
    devices = jax.local_devices()

    if cfg.rollout.decoder_pmap_axis == "batch":
        # decoding is parallelised over the batch --> every agent needs to be on every device.
        params = jax.device_put_replicated(params, devices)

        # send all the behavior markers to all the devices
        behavior_markers = jax.device_put_replicated(behavior_markers, devices)

    elif cfg.rollout.decoder_pmap_axis == "pop":
        # decoding is parallelised over the population --> spread the agents over the devices
        assert (
                cfg.training_sample_size >= cfg.num_devices
        ), f"Population of size {cfg.training_sample_size} too small for distribution over {cfg.num_devices} devices."
        assert (
                cfg.training_sample_size % cfg.num_devices == 0
        ), f"Population of size {cfg.training_sample_size} isn't divisibile by number of devices ({cfg.num_devices})."

        def distribute_params(p):
            shp = p.shape

            # split the parameters to put them on the different devices
            p = list(p.reshape(cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]))

            return jax.device_put_sharded(p, devices)

        # distribute
        params = jax.tree_map(distribute_params, params)
    else:
        raise ValueError(
            f"config.rollout.decoder_pmap_axis of {cfg.rollout.decoder_pmap_axis} not recognised"
        )

    if not keys:
        keys = list(random.split(training_key, cfg.num_devices))

    if not optimizer_state:
        # init the optimizer state
        optimizer_state = get_optimizer(cfg.optimizer).init(
            fetch_from_first_device(params)
        )

    training_state = TrainingState(
        params=params,
        behavior_markers=behavior_markers,
        optimizer_state=jax.device_put_replicated(optimizer_state, devices),
        num_steps=jax.device_put_replicated(num_steps, devices),
        key=jax.device_put_sharded(keys, devices),
        extras=jax.device_put_replicated(extras, devices),
    )

    return training_state


def generate_trajectory(
        networks: JobShopNetworks,
        environment,
        problem,
        params,
        behavior_marker,
        acting_key,
        stochastic=False,
):
    """Decode a single agent, from a single starting position on a single problem.

    With decorators, the expected input dimensions are:
        problems: [N, problem_size, 2]
        embeddings: [N, problem_size, 128]
        decoder_params (decoder only): {key: [...]} - only one! (the conditioned decoder)
        behavior_marker: [K, K]
        start_position: [N, K, M]
        acting_key: [N, K, M, 2]
    """

    def policy(
            observation: ObservationJobShop,
            key,
    ) -> Tuple[Array, Array, Array]:
        static_emb = jnp.zeros(
            (1, observation.problem.shape[1] + observation.problem.shape[2] + 1, 1), float)
        observation = jax.tree_map(lambda x: x[None], observation)

        actor_embeddings = networks.actor_encoder.apply(params.actor_encoder, observation)
        logits = networks.actor_decoder.apply(params.actor_decoder, actor_embeddings, static_emb,
                                              behavior_marker.reshape(1, -1),
                                              observation.action_mask)

        if stochastic:
            raw_action = networks.parametric_action_distribution.sample_no_postprocessing(
                logits, key
            )
        else:
            raw_action = networks.parametric_action_distribution.mode_no_postprocessing(logits)

        logprob = networks.parametric_action_distribution.log_prob(logits, raw_action)
        action = networks.parametric_action_distribution.postprocess(raw_action)

        critic_embeddings = networks.critic_encoder.apply(params.critic_encoder, observation)
        value = networks.critic_decoder.apply(params.critic_decoder, critic_embeddings, static_emb,
                                              behavior_marker.reshape(1, -1))

        action = action.reshape(-1)

        return action, logprob[0], value[0]

    def take_step(acting_state):
        key, act_key = random.split(acting_state.key, 2)

        # take action
        action, logprob, value = policy(acting_state.timestep.observation, act_key)

        # step in the environment
        state, timestep = environment.step(acting_state.state, action)
        info = Information(
            extras={"logprob": logprob, "value": value}, metrics={}, logging={}
        )

        # update the acting state
        acting_state = ActingState(state=state, timestep=timestep, key=key)

        return acting_state, (timestep, info)

    # reset
    state, timestep = environment.reset_from_state(problem)

    # create acting state
    acting_state = ActingState(state=state, timestep=timestep, key=acting_key)

    # scan steps in the env on a given horizon
    acting_state, (traj, info) = jax.lax.scan(
        lambda acting_state, _: take_step(acting_state),
        acting_state,
        xs=None,
        length=environment.get_episode_horizon(),
    )

    return acting_state, (traj, info)


def rollout(
        cfg: omegaconf.DictConfig,
        environment: PoppyJobShop,
        params: JobShopParams,
        behavior_markers: chex.Array,
        networks: JobShopNetworks,
        problems: jnp.ndarray,
        acting_keys: jnp.ndarray,
        stochastic: bool = False,
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
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """

    @functools.partial(jax.vmap, in_axes=(0, None, None, 0))  # over problems
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))  # over behaviour markers
    def generate_trajectory_fn(
            problem,
            params,
            behavior_marker,
            acting_key,
    ):
        return generate_trajectory(
            networks,
            environment,
            problem,
            params,
            behavior_marker,
            acting_key,
            stochastic,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        params,
        behavior_markers,
        acting_keys,
    )

    return acting_state, (traj, info)


def calculate_loss(
        traj, info, c_crit, use_poppy=False, use_poppy_hard=False
) -> chex.Array:
    """Calculate the loss of our population.

    For each (problem, instance) couple, train the strictly best agent (i.e., best latent vector).
    If there is a tie, no agent is trained.
    """

    # reward scaling
    reward = traj.reward / 100.0

    # returns to go
    returns_to_go = jnp.cumsum(reward[..., ::-1], axis=-1)[..., ::-1]  # [N, K, t]

    # extract the logprobs of all transitions
    logprob = info.extras["logprob"]  # [N, K, t]

    # Calculate advantages.
    advantages = returns_to_go - info.extras["value"]  # [N, K, t]

    # get best over agents - first one when there is a tie
    returns = reward.sum(-1, keepdims=True)  # [N, K, 1] returns_to_go[..., :1]
    train_idxs = returns.argmax(axis=1, keepdims=True)  # [N, 1, 1]

    max_reached = returns.max(axis=1, keepdims=True) == returns
    num_max_reached = jnp.sum(max_reached, axis=1, keepdims=True)
    mask = num_max_reached == 1  # [N, 1]

    # get values for the best agent only
    advantages = jnp.take_along_axis(advantages, train_idxs, axis=1)
    logprob = jnp.take_along_axis(logprob, train_idxs, axis=1)


    policy_loss = -jnp.mean(jax.lax.stop_gradient(advantages) * logprob, where=mask)
    critic_loss = jnp.mean(advantages ** 2, where=mask)
    loss = policy_loss + c_crit * critic_loss

    return jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


class Trainer:
    def __init__(
            self,
            cfg: omegaconf.DictConfig,
            logger,
    ):
        """Init a few elements and define the way n grad steps are applied."""

        self.cfg = cfg
        self.logger = logger
        self.environment = hydra.utils.instantiate(cfg.environment)
        self.networks = get_networks(cfg, self.environment)
        create_checkpoint_directory(cfg, self.logger)
        self.training_state = init_training_state(cfg, self.networks, self.environment)
        self.cfg.validation.num_devices = self.cfg.num_devices

        def sgd_step(training_state):
            # Prepare batch of problems, start positions and acting keys.
            key, problem_key = random.split(training_state.key, 2)
            num_problems = self.cfg.batch_size // self.cfg.num_devices
            num_agents = self.cfg.training_sample_size

            # prepare set of instances and starting positions
            problems, _, acting_keys = prepare_problem_batch(
                prng_key=problem_key,
                environment=self.environment,
                num_problems=num_problems,
                num_agents=num_agents,
                num_start_positions=-1
            )

            behavior_markers = cfg.behavior_amplification * jax.random.uniform(
                problem_key, shape=training_state.behavior_markers.shape, minval=-1, maxval=1
            )

            state, (traj, info) = rollout(
                cfg=self.cfg.rollout,
                environment=self.environment,
                params=training_state.params,
                behavior_markers=behavior_markers,
                networks=self.networks,
                problems=problems,
                acting_keys=acting_keys,
                stochastic=True,
            )

            # define fn to rollout and compute loss
            def loss_and_output(params, behavior_markers, problems, acting_keys, all_rewards):
                # rollout all agents on all observations
                state, (traj, info) = rollout(
                    cfg=self.cfg.rollout,
                    environment=self.environment,
                    params=params,
                    behavior_markers=behavior_markers,
                    networks=self.networks,
                    problems=problems,
                    acting_keys=acting_keys,
                    stochastic=True
                )

                # Mask logprob's for steps where the environement was done.
                #  - traj.observation.is_done = [0,0,...,0,1,1,...] with the first 1 at the terminal step.
                #  - we want to mask everything *after* the last step, hence the roll & setting the
                #    first step to always (obviously) not be done.
                is_done = (
                    jnp.roll(traj.observation.is_done, 1, axis=-1).at[..., 0].set(0)
                )
                info.extras["logprob"] *= 1 - is_done
                info.extras["value"] *= 1 - is_done

                strictly_best = (traj.reward.sum(-1).flatten() == all_rewards).sum() <= 1

                loss = jax.lax.cond(strictly_best,
                                    lambda _: calculate_loss(
                                        traj,
                                        info,
                                        c_crit=self.cfg.c_crit,
                                        use_poppy=self.cfg.use_poppy_objective,
                                        use_poppy_hard=self.cfg.use_poppy_hard_objective,
                                    ),
                                    lambda _: 0.0,
                                    operand=None)

                # Log loss and returns.
                info.metrics["loss"] = loss

                episode_return = traj.reward.sum(-1)  # [N, K]
                if self.environment.is_reward_negative():
                    ret_sign = -1
                else:
                    ret_sign = 1

                return_str = self.environment.get_reward_string()

                # store metrics
                info.metrics[f"{return_str}"] = (
                        ret_sign * episode_return.max(-1).mean()
                )
                info.metrics[f"{return_str}_rand_agent+start"] = (
                        ret_sign * all_rewards.mean()
                )

                # mean of completed episodes
                complete_episode_mask = jnp.where(
                    episode_return.reshape(-1) * ret_sign < self.environment.get_episode_horizon(),
                    True, False)
                return_complete_episode = jax.lax.cond(complete_episode_mask.all() == False,
                                                       lambda _: jnp.array(
                                                           self.environment.get_episode_horizon(),
                                                           float),
                                                       lambda _: (ret_sign * episode_return.reshape(
                                                           -1).mean(where=complete_episode_mask)),
                                                       None)
                info.metrics[f"{return_str}_complete_episodes"] = return_complete_episode

                incomplete = (episode_return.reshape(
                    -1) * ret_sign == 2 * self.environment.get_episode_horizon()).mean()
                invalid = (episode_return.reshape(
                    -1) * ret_sign > 2 * self.environment.get_episode_horizon()).mean()
                complete = (episode_return.reshape(
                    -1) * ret_sign <= self.environment.get_episode_horizon()).mean()

                info.metrics[f"incomplete_episodes"] = incomplete
                info.metrics[f"invalid_episodes"] = invalid
                info.metrics[f"complete_episodes"] = complete

                return loss, (state, (traj, info))

            params = training_state.params
            optimizer_state = training_state.optimizer_state

            rewards = traj.reward.sum(-1).flatten()
            agent_idx = rewards.argmax(-1)

            # get gradients
            grads, (_state, (_traj, info)) = jax.grad(loss_and_output, has_aux=True, )(
                params,
                behavior_markers[agent_idx][None],
                problems,
                acting_keys[:, agent_idx][None],
                rewards
            )

            if self.cfg.num_devices > 1:
                # Taking the mean across all devices to keep params in sync.
                grads = jax.lax.pmean(grads, axis_name="i")

            updates, optimizer_state = get_optimizer(self.cfg.optimizer).update(
                grads, optimizer_state, params=params
            )

            # update parameters
            params = optax.apply_updates(params, updates)

            # create new training state
            training_state = TrainingState(
                params=params,
                behavior_markers=training_state.behavior_markers,
                optimizer_state=optimizer_state,
                key=key,
                num_steps=training_state.num_steps + 1,
                extras=training_state.extras,
            )

            return training_state, info.metrics

        @functools.partial(jax.pmap, axis_name="i")
        def n_sgd_steps(training_state):
            # apply sequentially nb of steps
            training_state, metrics = jax.lax.scan(
                lambda state, xs: sgd_step(state),
                init=training_state,
                xs=None,
                length=self.cfg.num_jit_steps,
            )

            # Average metrics over all jit-ted steps.
            metrics = jax.tree_map(lambda x: x.mean(0), metrics)

            return training_state, metrics

        self.n_sgd_steps = n_sgd_steps

    def train(self):  # noqa: CCR001
        """Main method of the trainer class. Handle the training of compass."""

        random_key = jax.random.PRNGKey(self.cfg.seed)

        def get_n_steps():
            if self.cfg.num_devices > 1:
                n_steps = fetch_from_first_device(self.training_state.num_steps)
            else:
                n_steps = self.training_state.num_steps
            return n_steps

        def log(metrics, key=None):
            metrics["step"] = get_n_steps()
            if self.logger:
                if key:
                    metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}
                self.logger.write(metrics)

        # update mixed precision mode
        set_train_policy = lambda: set_policy(
            modules=[hk.MultiHeadAttention],
            use_half=self.cfg.use_half_precision,
        )
        set_validation_policy = lambda: set_policy(
            modules=[hk.MultiHeadAttention],
            use_half=self.cfg.validation.use_half_precision,
        )

        set_train_policy()

        # main loop of the method
        while get_n_steps() <= self.cfg.num_steps:
            # do validation step (under condition)
            if get_n_steps() % self.cfg.validation_freq == 0:
                set_validation_policy()
                t = time.time()

                # fetch the training state
                training_state = fetch_from_first_device(self.training_state)

                # compute validation metrics
                random_key, subkey = jax.random.split(random_key)
                metrics = validation.validate(
                    subkey,
                    self.cfg.validation,
                    training_state.params,
                    behavior_dim=self.cfg.behavior_dim,
                )
                jax.tree_map(
                    lambda x: x.block_until_ready(), metrics
                )  # For accurate timings.
                metrics["total_time"] = time.time() - t
                if self.cfg.num_devices > 1:
                    metrics = reduce_from_devices(metrics, axis=0)
                log(metrics, "validate")

                set_train_policy()

                reward_str = self.environment.get_reward_string()

                # save checkpoints
                if self.cfg.checkpointing.save_checkpoint:
                    training_state = fetch_from_first_device(
                        self.training_state
                    ).replace(key=self.training_state.key)
                    save_checkpoint(
                        self.cfg,
                        training_state,
                        self.logger,
                    )

                    if (
                            metrics[reward_str] > training_state.extras["best_reward"]
                            and self.cfg.checkpointing.keep_best_checkpoint
                    ):
                        save_checkpoint(
                            self.cfg,
                            training_state,
                            self.logger,
                            fname_prefix="best_",
                        )

                        extras = self.training_state.extras
                        extras.update(
                            {
                                "best_reward": jnp.ones_like(extras["best_reward"])
                                               * metrics[reward_str]
                            }
                        )

                        self.training_state = TrainingState(
                            params=self.training_state.params,
                            behavior_markers=self.training_state.behavior_markers,
                            optimizer_state=self.training_state.optimizer_state,
                            num_steps=self.training_state.num_steps,
                            key=self.training_state.key,
                            extras=extras,
                        )

                    print(f"Saved checkpoint at step {get_n_steps()}")
            if (
                    get_n_steps() % self.cfg.checkpointing.intermediate_saving_frequency
                    == 0
            ):
                # save the checkpoints at regular intervals
                if self.cfg.checkpointing.save_checkpoint:
                    training_state = fetch_from_first_device(
                        self.training_state
                    ).replace(key=self.training_state.key)
                    save_checkpoint(
                        self.cfg,
                        training_state,
                        self.logger,
                        fname_prefix=f"step_{get_n_steps()}_",
                    )

            t = time.time()

            # apply n gradient steps - training is here
            self.training_state, metrics = self.n_sgd_steps(self.training_state)

            jax.tree_map(
                lambda x: x.block_until_ready(), metrics
            )  # For accurate timings.

            if self.cfg.num_devices > 1:
                metrics = reduce_from_devices(metrics, axis=0)

            metrics["step_time"] = (time.time() - t) / self.cfg.num_jit_steps

            # handle the logging
            log(metrics, "train")
