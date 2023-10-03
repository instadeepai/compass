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
from typing import Any, Optional, Tuple

import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf
from jumanji.types import TimeStep

import compass.trainers.routing.trainer_routing as trainer
from compass.environments.poppy_env import PoppyEnv
from compass.networks import Networks
from compass.trainers.trainer_utils import ActingState, Information
from compass.trainers.routing.trainer_routing import generate_trajectory
from compass.trainers.validation_utils import get_instances, get_params
from compass.utils.data import get_acting_keys
from compass.utils.emitter_pool import CMAPoolEmitter
from compass.utils.metrics import get_metrics
from compass.utils.utils import spread_over_devices


def slowrl_rollout(
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

    Returns:
        # TODO
    """

    # split the params in encoder and decoder - those a merged in the training state
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # initialise the embeddings for each problem
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0, 0))  # over N problems
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, 0, 0, 0)
    )  # over K agents - behaviors
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, None, 0, 0)
    )  # M starting pos.
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


def slowrl_validate(
    random_key,
    cfg: omegaconf.DictConfig,
    params: chex.ArrayTree = None,
    behavior_dim: Optional[int] = None,
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

    if cfg.rollout.decoder_pmap_axis == "pop":
        # TODO: Handle metric collection in this case.
        raise NotImplementedError

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, start_positions, acting_keys, behavior_markers):
        """Run the rollout on a batch of problems and return the episode return.
        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """
        # split problems, start_positions and acting_keys into chunks of size batch_size.
        num_batches = int(round(len(problems) / cfg.batch_size, 0))

        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        start_positions = jnp.stack(jnp.split(start_positions, num_batches, axis=0))
        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))
        behavior_markers = jnp.stack(jnp.split(behavior_markers, num_batches, axis=0))

        num_problems = problems.shape[1]

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)

            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )

            # Note, the starting positions and acting keys are duplicated here.
            start_positions = jnp.repeat(start_positions, 8, axis=1)
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)
            behavior_markers = jnp.repeat(behavior_markers, 8, axis=1)

        def body(_, x):
            problems, start_positions, acting_keys, behavior_markers = x
            _, (traj, info) = slowrl_rollout(
                cfg=cfg.rollout,
                environment=environment,
                params=params,
                behavior_markers=behavior_markers,
                networks=networks,
                problems=problems,
                start_positions=start_positions,
                acting_keys=acting_keys,
            )
            info.metrics["rewards"] = traj.reward
            return None, info.metrics

        _, metrics = jax.lax.scan(
            body,
            init=None,
            xs=(problems, start_positions, acting_keys, behavior_markers),
        )

        if cfg.use_augmentations:
            num_agents, num_start_positions = (
                start_positions.shape[-2],
                start_positions.shape[-1],
            )
            metrics = jax.tree_map(
                lambda x: x.reshape(
                    num_batches,
                    num_problems,
                    8,
                    num_agents,
                    num_start_positions,
                    -1,
                ).max(
                    2
                ),  # max on the pb augmentation dimension
                metrics,
            )

        # flatten batch dimension of metrics
        metrics = jax.tree_map(lambda x: x.reshape(*(-1,) + x.shape[2:]), metrics)
        episode_return = metrics["rewards"].sum(-1)  # [N, K, M]

        return episode_return

    # instantiate networks and environments
    networks = trainer.get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)
    if not params:
        params = get_params(cfg.checkpointing)

    # define the number of starting points
    if cfg.num_starting_points < 0:
        num_starting_points = environment.get_problem_size()
    else:
        num_starting_points = cfg.num_starting_points

    # get a set of instances
    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
        pop_size=cfg.validation_pop_size,
    )

    # from now one, we want to use optimally a given budget
    print("Num starting points: ", num_starting_points)
    budget = cfg.budget * num_starting_points

    print("Budget: ", budget)

    # create behavior markers
    random_key, subkey = jax.random.split(random_key)
    noshard_behavior_markers = cfg.behavior_amplification * jax.random.uniform(
        subkey,
        shape=(cfg.problems.num_problems, cfg.validation_pop_size, behavior_dim),
        minval=-1,
        maxval=1,
    )

    # replicate them over the devices
    devices = jax.local_devices()
    # get shape
    shp = noshard_behavior_markers.shape

    # split the parameters to put them on the different devices
    behavior_markers = list(
        noshard_behavior_markers.reshape(
            cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]
        )
    )

    behavior_markers = jax.device_put_sharded(behavior_markers, devices)

    init_sigma = cfg.cmaes_sigma

    random_key, subkey = jax.random.split(random_key)
    emitter = CMAPoolEmitter(
        num_states=cfg.num_cmaes_states,
        population_size=cfg.validation_pop_size,
        num_best=cfg.validation_pop_size // 2,
        search_dim=behavior_dim,
        init_sigma=float(cfg.cmaes_sigma),
        delay_eigen_decomposition=False,
        init_minval=-cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        init_maxval=cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        random_key=subkey,
    )

    emitter_state = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.expand_dims(x, axis=0), repeats=cfg.problems.num_problems, axis=0
        ),
        emitter.init(),
    )

    # put while loop here - keep all values
    used_budget = 0
    best_episode_return = None

    use_poppy_strategy = cfg.use_poppy_strategy
    update_behavior_markers = cfg.update_behavior_markers
    use_cmaes = cfg.use_cmaes

    if cfg.strategy is not None:
        # overwrite the params by the ones defining the given strategy
        if cfg.strategy == "naive-rollouts":
            update_behavior_markers = False
            use_cmaes = False
            use_poppy_strategy = False
        elif cfg.strategy == "greedy-rollouts":
            update_behavior_markers = False
            use_cmaes = False
            use_poppy_strategy = True
        elif cfg.strategy == "behavior-sampling":
            update_behavior_markers = True
            use_cmaes = False
            use_poppy_strategy = False
        elif cfg.strategy == "behavior-cmaes":
            update_behavior_markers = True
            use_cmaes = True
            use_poppy_strategy = False

    while used_budget < budget:
        # update the acting keys - for stochasticity
        new_acting_keys = cfg.new_acting_keys
        if new_acting_keys:
            random_key, subkey = jax.random.split(random_key)
            acting_keys = get_acting_keys(
                subkey,
                num_starting_points,
                cfg.problems.num_problems,
                cfg.validation_pop_size,
            )
            acting_keys = spread_over_devices(acting_keys, devices=devices)

        if update_behavior_markers:
            if use_cmaes:
                random_key, subkey = jax.random.split(random_key)
                subkeys = jax.random.split(subkey, num=cfg.problems.num_problems)
                behavior_markers, _random_keys = jax.vmap(emitter.sample)(
                    emitter_state, subkeys
                )

                noshard_behavior_markers = behavior_markers

            else:
                random_key, subkey = jax.random.split(random_key)
                behavior_markers = cfg.behavior_amplification * jax.random.uniform(
                    subkey,
                    shape=(
                        cfg.problems.num_problems,
                        cfg.validation_pop_size,
                        behavior_dim,
                    ),
                    minval=-1,
                    maxval=1,
                )

        if update_behavior_markers:
            # get shape
            shp = behavior_markers.shape

            # split the parameters to put them on the different devices
            behavior_markers = list(
                behavior_markers.reshape(
                    cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]
                )
            )

            behavior_markers = jax.device_put_sharded(behavior_markers, devices)

        # run the validation episodes
        episode_return = run_validate(
            problems, start_positions, acting_keys, behavior_markers
        )
        episode_return = jnp.concatenate(episode_return, axis=0)

        if use_cmaes:
            # sort behavior markers based on the perf we got
            fitnesses = -episode_return.max(-1)

            # only take the actor (hence behavior marker) into account
            sorted_indices = jnp.argsort(fitnesses, axis=-1)

            # sort the behaviors accordingly
            sorted_behavior_markers = jax.vmap(functools.partial(jnp.take, axis=0))(
                noshard_behavior_markers, sorted_indices
            )

            # use it to update the state
            emitter_state = jax.vmap(emitter.update_state)(
                emitter_state,
                sorted_candidates=sorted_behavior_markers[
                    :, : cfg.validation_pop_size // 2
                ],
            )

        if use_poppy_strategy and used_budget == 0:
            # flatten all the returns obtained (for each episode)
            problems_episode_return = jax.vmap(jnp.ravel)(episode_return)  # [N, K*M]

            # sort agent/starting points pairs from the first evaluation
            problems_sorted_indices = jax.vmap(jnp.argsort)(
                -problems_episode_return
            )  # [N, K*M]

            # unravel those indices - (gives them asif the returns had not been flattened)
            sorted_indices = jax.vmap(
                functools.partial(jnp.unravel_index, shape=episode_return.shape[-2:])
            )(
                problems_sorted_indices
            )  # ([N, K*M], [N, K*M])

            # get the sorted indices
            sorted_indices_agent, sorted_indices_start_position = sorted_indices

            max_length = sorted_indices_agent.shape[1]

            # extract only the K best out of K*M
            sorted_indices_agent = sorted_indices_agent[
                :, : cfg.validation_pop_size, ...
            ]  # [N, K]
            sorted_indices_start_position = sorted_indices_start_position[
                :, : cfg.validation_pop_size, ...
            ]  # [N, K]

            repeats = int(jnp.ceil(max_length / cfg.validation_pop_size))  # = M

            # repeat best starting positions to get the same number of rollouts as before
            sorted_indices_start_position_repeated = jnp.repeat(
                sorted_indices_start_position, repeats=repeats, axis=1
            )[
                :, :max_length, ...
            ]  # make sure not to overlap - [N, K*M]

            # put start position in same shape as the sorted indices
            noshard_start_positions = jnp.concatenate(list(start_positions), axis=0)
            flat_start_positions = jax.vmap(jnp.ravel)(noshard_start_positions)

            # functions to extract the corresponding agents and starting points
            take_array_indices_1 = lambda arr, x: arr[x, ...]
            take_array_indices_2 = lambda arr, x: arr[x]

            # extract the best behavior descriptors
            desired_behavior_descriptors = jax.vmap(take_array_indices_1)(
                noshard_behavior_markers, sorted_indices_agent
            )  # [N, K] (with non unique behavior markers)

            # extract the starting points that got the best perfs (w/ those behaviors)
            desired_start_positions = jax.vmap(take_array_indices_2)(
                flat_start_positions,
                sorted_indices_start_position_repeated,
            )  # [N, K*M] (with repeated starting positions)

            # reshape start position
            desired_start_positions = jnp.reshape(
                desired_start_positions, noshard_start_positions.shape
            )  # [N, K, M]

            # re-arrange for devices
            # get shape
            shp = desired_behavior_descriptors.shape

            # split the parameters to put them on the different devices
            behavior_markers = list(
                desired_behavior_descriptors.reshape(
                    cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]
                )
            )

            # get final behavior and starting point that will be used til the
            # end of the given budget
            behavior_markers = jax.device_put_sharded(
                behavior_markers, devices
            )  # [D, N/D, K, M]
            start_positions = spread_over_devices(
                desired_start_positions
            )  # [D, N/D, K, M]

        latest_batch_best = episode_return.max((-1, -2))
        if used_budget == 0:
            best_episode_return = latest_batch_best
        else:
            # get latest best

            best_episode_return = jnp.concatenate(
                [best_episode_return[:, None], latest_batch_best[:, None]], axis=1
            ).max(-1)

        if environment.is_reward_negative():
            ret_sign = -1
        else:
            ret_sign = 1
        return_str = environment.get_reward_string()

        # get latest batch min, mean, max and std
        latest_batch_best_sp = episode_return.max(-1)
        latest_batch_min = latest_batch_best_sp.min(-1)
        latest_batch_mean = latest_batch_best_sp.mean(-1)
        latest_batch_std = latest_batch_best_sp.std(-1)

        # Make new metrics dictionary which will be all the returned statistics.
        metrics = {
            f"{return_str}_latest_batch": ret_sign * latest_batch_best.mean(),
            f"{return_str}": ret_sign * best_episode_return.mean(),
            f"{return_str}_latest_min": ret_sign * latest_batch_min.mean(),
            f"{return_str}_latest_mean": ret_sign * latest_batch_mean.mean(),
            f"{return_str}_latest_std": latest_batch_std.mean(),
        }

        metrics = get_metrics(
            metrics,
            episode_return,
            compute_expensive_metrics=cfg.compute_expensive_metrics,
        )

        # update the used budget
        used_budget += cfg.validation_pop_size * num_starting_points
        print("Used budget: ", used_budget)

        log(metrics, used_budget, logger, "slowrl")

    return metrics
