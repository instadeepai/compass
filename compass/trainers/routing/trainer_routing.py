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
import acme
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import Array
from jax import random
from jumanji.types import TimeStep

import compass.trainers.routing.validation_routing as validation
from compass.environments import PoppyEnv
from compass.networks import DecoderBase, EncoderBase, Networks
from compass.utils import (
    create_checkpoint_directory,
    load_checkpoint,
    save_checkpoint,
    prepare_problem_batch,
    get_layers_with_offset_names,
    sync_params_and_offset,
    fetch_from_first_device,
    generate_zeros_from_spec,
    reduce_from_devices,
)
from compass.trainers.trainer_utils import (
    Observation,
    Information,
    ActingState,
    TrainingStateRouting as TrainingState,
    set_policy,
)


def get_optimizer(cfg: omegaconf.DictConfig) -> optax.GradientTransformation:
    """Create desired optimizer"""
    # TODO: not sure what this is for yet
    encoder_mask_fn = functools.partial(
        hk.data_structures.map, lambda m, n, p: "encoder" in m
    )
    decoder_mask_fn = functools.partial(
        hk.data_structures.map, lambda m, n, p: "encoder" not in m
    )

    # define optimizer - encoder/decoder
    optimizer = optax.chain(
        optax.masked(
            optax.adamw(
                learning_rate=cfg.encoder.lr,
                weight_decay=cfg.encoder.l2_regularization,
            ),
            encoder_mask_fn,
        ),
        optax.masked(
            optax.adamw(
                learning_rate=cfg.decoder.lr,
                weight_decay=cfg.decoder.l2_regularization,
            ),
            decoder_mask_fn,
        ),
    )

    # use wrapper for gradient accumulation
    optimizer = optax.MultiSteps(optimizer, cfg.num_gradient_accumulation_steps)

    return optimizer


def get_networks(cfg) -> Networks:
    """Create networks used in the training."""

    def encoder_fn(problem: chex.Array):
        # use the class/params given in config to instantiate an encoder
        encoder = hydra.utils.instantiate(cfg.encoder, name="shared_encoder")
        return encoder(problem)

    def decoder_fn(
        observation: Observation,
        embeddings: Array,
        behavior_marker: Array,
        condition_query: bool,
        condition_key: bool,
        condition_value: bool,
    ):
        # use the class/params given in config to instantiate decoder
        decoder = hydra.utils.instantiate(cfg.decoder, name="decoder")

        return decoder(
            observation,
            embeddings,
            behavior_marker,
            condition_query,
            condition_key,
            condition_value,
        )

    return Networks(
        encoder_fn=hk.without_apply_rng(hk.transform(encoder_fn)),
        decoder_fn=hk.without_apply_rng(hk.transform(decoder_fn)),
    )


def init_training_state(
    cfg: omegaconf.DictConfig, networks: Networks, environment: PoppyEnv
) -> TrainingState:
    """Instantiate the initial state of the training process."""

    key = random.PRNGKey(cfg.seed)
    encoder_key, decoder_key, training_key = random.split(key, 3)

    # load checkpoints
    (
        encoder_params,
        decoder_params,
        optimizer_state,
        keys,
        num_steps,
        extras,
    ) = load_checkpoint(cfg)

    decoder_params = jax.tree_util.tree_map(lambda x: x[0], decoder_params)

    # retrieve spec of the environment
    environment_spec = acme.make_environment_spec(environment)

    # dummy obs used to create the models params
    _dummy_obs = environment.make_observation(
        *jax.tree_map(
            generate_zeros_from_spec,
            environment_spec.observations.generate_value(),
        )
    )

    if not encoder_params:
        encoder_params = networks.encoder_fn.init(encoder_key, _dummy_obs.problem)

    # create the conditioned decoder from the learned encoder
    conditioned_decoder_params = None
    if decoder_params is not None:
        # needed to create the decoders params
        _dummy_embeddings = networks.encoder_fn.apply(
            encoder_params, _dummy_obs.problem
        )

        _dummy_behavior_marker = jnp.zeros(shape=(cfg.behavior_dim))

        # init conditioned_decoder_params
        decoder_conditions = cfg.rollout.decoder_conditions
        conditioned_decoder_params = networks.decoder_fn.init(
            decoder_key,
            _dummy_obs,
            _dummy_embeddings,
            _dummy_behavior_marker,
            decoder_conditions.query,
            decoder_conditions.key,
            decoder_conditions.value,
        )

        layers_names = get_layers_with_offset_names(decoder_conditions)

        print("Layers names: ", layers_names)

        conditioned_decoder_params = sync_params_and_offset(
            conditioned_decoder_params,
            decoder_params,
            offset_size=cfg.behavior_dim,
            layers_names=layers_names,
        )

    # define the behavior markers
    behavior_markers = jnp.zeros((cfg.training_sample_size, cfg.behavior_dim))

    if not keys:
        keys = list(random.split(training_key, cfg.num_devices))

    # distribute parameters over devices as required.
    devices = jax.local_devices()

    # put the encoder on every device
    encoder_params = jax.device_put_replicated(encoder_params, devices)

    if cfg.rollout.decoder_pmap_axis == "batch":
        # decoding is parallelised over the batch --> every agent needs to be on every device.
        decoder_params = jax.device_put_replicated(decoder_params, devices)
        conditioned_decoder_params = jax.device_put_replicated(
            conditioned_decoder_params, devices
        )

        # send all the behavior markers to all the devices
        # print("Behavior markers: ", behavior_markers.shape)
        behavior_markers = jax.device_put_replicated(behavior_markers, devices)
        # print("Behavior markers: ", behavior_markers.shape)

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
        decoder_params = jax.tree_map(distribute_params, decoder_params)
    else:
        raise ValueError(
            f"config.rollout.decoder_pmap_axis of {cfg.rollout.decoder_pmap_axis} not recognised"
        )

    # merge the params dict into a single one
    params = hk.data_structures.merge(encoder_params, conditioned_decoder_params)

    # if not created from the checkpoints - init the optimizer
    if not optimizer_state:
        # init the optimizer state
        optimizer_state = get_optimizer(cfg.optimizer).init(
            fetch_from_first_device(params)
        )

    training_state = TrainingState(
        params=params,
        decoder_params=decoder_params,
        behavior_markers=behavior_markers,  # has already been duplicated on the devices
        optimizer_state=jax.device_put_replicated(optimizer_state, devices),
        num_steps=jax.device_put_replicated(num_steps, devices),
        key=jax.device_put_sharded(keys, devices),
        extras=jax.device_put_replicated(extras, devices),
    )

    return training_state


def generate_trajectory(
    decoder_conditions,
    decoder_apply_fn,
    policy_temperature,
    environment,
    problem,
    embeddings,
    decoder_params,
    behavior_marker,
    start_position,
    acting_key,
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
        observation: Observation,
        key,
    ) -> Array:
        # get logits from the decoder

        logits = decoder_apply_fn(
            decoder_params,
            observation,
            embeddings,
            behavior_marker,
            decoder_conditions.query,
            decoder_conditions.key,
            decoder_conditions.value,
        )
        logits -= 1e30 * observation.action_mask

        # take action
        if policy_temperature > 0:
            action = rlax.softmax(temperature=policy_temperature).sample(key, logits)
        else:
            action = rlax.greedy().sample(key, logits)

        # compute the logprob - used by Reinforce
        logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)

        return action, logprob

    def take_step(acting_state):
        key, act_key = random.split(acting_state.key, 2)

        # take action
        action, logprob = policy(acting_state.timestep.observation, act_key)

        # step in the environment
        state, timestep = environment.step(acting_state.state, action)
        info = Information(extras={"logprob": logprob}, metrics={}, logging={})

        # update the acting state
        acting_state = ActingState(state=state, timestep=timestep, key=key)

        return acting_state, (timestep, info)

    # reset
    state, timestep = environment.reset_from_state(problem, start_position)

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
    environment: PoppyEnv,
    params: chex.ArrayTree,
    behavior_markers: chex.Array,
    networks: Networks,
    problems: jnp.ndarray,
    start_positions: jnp.ndarray,
    acting_keys: jnp.ndarray,
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
    """

    # split the params in encoder and decoder - those a merged in the training state
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # initialise the embeddings for each problem
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    if cfg.decoder_pmap_axis == "pop" and cfg.encoder_pmap_axis == "batch":
        """
        If the encoder is distributed over a batch of N instances, each of the D
        devices encodes N/D problems.  However, if the decoding of K agents is
        distributed over the population, then each device should decode K/D agents
        on all N problems. Therefore, here we fetch the embeddings from all other
        devices.

        TODO: start_positions and acting_keys could of been generated on the correct
        devices as they aren't used for the encoding.
        """

        embeddings = jax.lax.all_gather(embeddings, "i", axis=0).reshape(
            -1, *embeddings.shape[1:]
        )
        problems = jax.lax.all_gather(problems, "i", axis=0).reshape(
            -1, *problems.shape[1:]
        )
        start_positions = jax.lax.all_gather(start_positions, "i", axis=0).reshape(
            -1, *start_positions.shape[1:]
        )
        acting_keys = jax.lax.all_gather(acting_keys, "i", axis=0).reshape(
            -1, *acting_keys.shape[1:]
        )

    if cfg.decoder_pmap_axis == "batch" and cfg.encoder_pmap_axis == "pop":
        raise NotImplementedError("This is an efficient configuration and thus, not implemented.")

    @functools.partial(jax.vmap, in_axes=(0, 0, None, None, 0, 0))  # over N problems
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0))  # over K agents - behaviors
    @functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0))  # M starting pos.
    def generate_trajectory_fn(
        problem,
        embeddings,
        decoder_params,
        behavior_marker,
        start_position,
        acting_key,
    ):
        return generate_trajectory(
            cfg.decoder_conditions,
            networks.decoder_fn.apply,
            cfg.policy.temperature,
            environment,
            problem,
            embeddings,
            decoder_params,
            behavior_marker,
            start_position,
            acting_key,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        embeddings,
        decoder_params,
        behavior_markers,
        start_positions,
        acting_keys,
    )

    return acting_state, (traj, info)


def calculate_loss(traj, info) -> chex.Array:
    """Calculate the loss of our population.

    For each (problem, instance) couple, train the strictly best agent (i.e., best latent vector).
    If there is a tie, no agent is trained.
    """
    # sum rewards over the traj
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]

    # extract and sum the logprobs over the traj
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # Calculate advantages.
    if returns.shape[-1] > 1:
        advantages = returns - returns.mean(-1, keepdims=True)  # [N, K, M]
    else:
        advantages = returns  # [N, K, M]

    # get best over agents - first one when there is a tie
    train_idxs = returns.argmax(axis=1, keepdims=True)  # [N, 1, M]

    # get values for the best agent only
    advantages = jnp.take_along_axis(advantages, train_idxs, axis=1)
    logprob_traj = jnp.take_along_axis(logprob_traj, train_idxs, axis=1)

    mask = jnp.where(
        (returns.max(axis=1, keepdims=True) == returns).sum(
            axis=1, keepdims=True
        )
        > 1,
        False,
        True,
    )  # [N, 1, M]
    loss = -jnp.mean(advantages * logprob_traj, where=mask)

    # replace nan with zeros - which is the case when they are always ties
    loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss


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
        self.networks = get_networks(cfg.networks)
        create_checkpoint_directory(cfg, self.logger)
        self.training_state = init_training_state(cfg, self.networks, self.environment)

        self.cfg.validation.num_devices = self.cfg.num_devices

        if (
            self.cfg.validation.use_augmentations
            and "Knapsack" in cfg.environment._target_
        ):
            raise ValueError(
                "Knapsack's problem instances cannot be augmented, set "
                "'use_augmentations' in config.validate to False."
            )

        def sgd_step(training_state):
            """Apply a gradient step.

            training_state contains:

            params: with the encoder and the conditioned decoder - each once only
            decoder_params: the decoder_params from the training phase
            behavior_markers: all the behavior markers (training_sample_size, behavior_dim)
            optimizer_state: optax.OptState
            num_steps: jnp.int32
            key: PRNGKey
            extras: Optional[dict] = field(default_factory=dict)
            """

            # define fn to rollout and compute loss
            def loss_and_output(
                params, behavior_markers, problems, start_positions, acting_keys
            ):
                # rollout all decoders on all pbs/starting points
                state, (traj, info) = rollout(
                    cfg=self.cfg.rollout,
                    environment=self.environment,
                    params=params,
                    behavior_markers=behavior_markers,
                    networks=self.networks,
                    problems=problems,
                    start_positions=start_positions,
                    acting_keys=acting_keys,
                )

                # Mask logprob's for steps where the environement was done.
                #  - traj.observation.is_done = [0,0,...,0,1,1,...] with the first 1 at the terminal step.
                #  - we want to mask everything *after* the last step, hence the roll & setting the
                #    first step to always (obviously) not be done.
                is_done = (jnp.roll(traj.observation.is_done, 1, axis=-1).at[..., 0].set(0))
                info.extras["logprob"] *= 1 - is_done

                # get loss
                loss = calculate_loss(traj, info)

                # Log loss and returns.
                info.metrics["loss"] = loss

                episode_return = traj.reward.sum(-1)  # [N, K, M]
                if self.environment.is_reward_negative():
                    ret_sign = -1
                else:
                    ret_sign = 1

                return_str = self.environment.get_reward_string()

                # store metrics
                info.metrics[f"{return_str}"] = (
                    ret_sign * episode_return.max((-1, -2)).mean()
                )
                if self.cfg.training_sample_size > 1:
                    info.metrics[f"{return_str}_rand_agent"] = (
                        ret_sign * episode_return.max(-1).mean()
                    )
                if self.cfg.num_starting_positions != 1:
                    info.metrics[f"{return_str}_rand_start"] = (
                        ret_sign * episode_return.max(-2).mean()
                    )
                if (self.cfg.training_sample_size > 1) and (
                    self.cfg.num_starting_positions != 1
                ):
                    info.metrics[f"{return_str}_rand_agent+start"] = (
                        ret_sign * episode_return.mean()
                    )

                return loss, (state, (traj, info))

            # Prepare batch of problems, start positions and acting keys.
            key, problem_key = random.split(training_state.key, 2)

            if self.cfg.rollout.encoder_pmap_axis == "pop":
                num_problems = self.cfg.batch_size
                duplicate_problems_on_each_device = True
            else:
                num_problems = self.cfg.batch_size // self.cfg.num_devices
                duplicate_problems_on_each_device = False

            if self.cfg.rollout.decoder_pmap_axis == "pop":
                num_agents = self.cfg.training_sample_size // self.cfg.num_devices
            else:
                num_agents = self.cfg.training_sample_size

            if duplicate_problems_on_each_device:
                # Distribute the problem key from the first device to all devices.
                problem_key = jax.lax.all_gather(problem_key, "i", axis=0)[0]

            # prepare set of instances and starting positions
            problems, start_positions, acting_keys = prepare_problem_batch(
                prng_key=problem_key,
                environment=self.environment,
                num_problems=num_problems,
                num_agents=num_agents,
                num_start_positions=self.cfg.num_starting_positions,
                duplicate_problems_on_each_device=duplicate_problems_on_each_device,
            )

            params = training_state.params
            optimizer_state = training_state.optimizer_state

            # sample behavior markers
            key, subkey = jax.random.split(key)
            behavior_markers = cfg.behavior_amplification * jax.random.uniform(
                subkey, shape=training_state.behavior_markers.shape, minval=-1, maxval=1
            )

            # get gradients
            grads, (_state, (_traj, info)) = jax.grad(
                loss_and_output,
                has_aux=True,
            )(
                params,
                behavior_markers,
                problems,
                start_positions,
                acting_keys,
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
                decoder_params=training_state.decoder_params,
                behavior_markers=behavior_markers,
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
            modules=[EncoderBase, DecoderBase, hk.MultiHeadAttention],
            use_half=self.cfg.use_half_precision,
        )
        set_validation_policy = lambda: set_policy(
            modules=[EncoderBase, DecoderBase, hk.MultiHeadAttention],
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
                            decoder_params=self.training_state.decoder_params,
                            behavior_markers=training_state.behavior_markers,
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
