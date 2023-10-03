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
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf
from jumanji.types import TimeStep

import compass.trainers.packing.trainer_packing as trainer
from compass.environments.poppy_env import PoppyEnv
from compass.networks.jobshop import JobShopNetworks
from compass.trainers.validation_utils import get_instances, get_params
from compass.utils import get_acting_keys, spread_over_devices
from compass.utils.emitter_pool import CMAPoolEmitter


def slowrl_rollout(
        cfg: omegaconf.DictConfig,
        environment: PoppyEnv,
        params: chex.ArrayTree,
        behavior_markers: chex.Array,
        networks: JobShopNetworks,
        problems: jnp.ndarray,
        acting_keys: jnp.ndarray,
) -> Tuple[trainer.ActingState, Tuple[TimeStep, trainer.Information]]:
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

    Returns:
        # TODO
    """

    @functools.partial(jax.vmap, in_axes=(0, None, 0, 0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))  # over K agents - behaviors
    def generate_trajectory_fn(
            problem,
            params,
            behavior_marker,
            acting_key,
    ):
        return trainer.generate_trajectory(
            networks,
            environment,
            problem,
            params,
            behavior_marker,
            acting_key,
            stochastic=True,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        params,
        behavior_markers,
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

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, acting_keys, behavior_markers):
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
        num_agents = acting_keys.shape[1]
        num_batches = int(round(len(problems) / cfg.batch_size, 0))

        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))
        behavior_markers = jnp.stack(jnp.split(behavior_markers, num_batches, axis=0))

        num_problems = problems.shape[1]

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)

            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )

            # Note, the starting positions and acting keys are duplicated here.
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)
            behavior_markers = jnp.repeat(behavior_markers, 8, axis=1)

        def body(_, x):
            problems, acting_keys, behavior_markers = x
            _, (traj, info) = slowrl_rollout(
                cfg=cfg.rollout,
                environment=environment,
                params=params,
                behavior_markers=behavior_markers,
                networks=networks,
                problems=problems,
                acting_keys=acting_keys,
            )
            info.metrics["rewards"] = traj.reward
            return None, info.metrics

        _, metrics = jax.lax.scan(
            body,
            init=None,
            xs=(problems, acting_keys, behavior_markers),
        )

        if cfg.use_augmentations:
            metrics = jax.tree_map(
                lambda x: x.reshape(
                    num_batches,
                    num_problems,
                    8,
                    num_agents,
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
    environment = hydra.utils.instantiate(cfg.environment)
    networks = trainer.get_networks(cfg, environment)
    if not params:
        params = get_params(cfg.checkpointing)

    # get a set of instances
    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
        pop_size=cfg.validation_pop_size,
        batch_size=cfg.batch_size,
    )

    # replicate them over the devices
    devices = jax.local_devices()

    # from now one, we want to use optimally a given budget
    budget = cfg.budget
    print("Budget: ", budget)
    num_problems = problems.shape[0] * problems.shape[1]

    # create behavior markers
    random_key, subkey = jax.random.split(random_key)
    noshard_behavior_markers = cfg.behavior_amplification * jax.random.uniform(
        subkey,
        shape=(num_problems, cfg.validation_pop_size, behavior_dim),
        minval=-1,
        maxval=1,
    )

    # get shape
    shp = noshard_behavior_markers.shape

    # split the parameters to put them on the different devices
    behavior_markers = list(
        noshard_behavior_markers.reshape(
            cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]
        )
    )

    behavior_markers = jax.device_put_sharded(behavior_markers, devices)

    random_key, subkey = jax.random.split(random_key)
    emitter = CMAPoolEmitter(
        num_states=cfg.num_cmaes_states,
        population_size=cfg.validation_pop_size,
        num_best=(cfg.validation_pop_size // 4) * 3,
        search_dim=behavior_dim,
        init_sigma=float(cfg.cmaes_sigma),
        delay_eigen_decomposition=False,
        init_minval=-cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        init_maxval=cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        random_key=subkey,
    )

    emitter_state = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.expand_dims(x, axis=0), repeats=num_problems, axis=0
        ),
        emitter.init(),
    )

    # put while loop here - keep all values
    used_budget = 0
    best_episode_return = None

    update_behavior_markers = cfg.update_behavior_markers
    use_cmaes = cfg.use_cmaes

    if cfg.strategy is not None:
        # overwrite the params by the ones defining the given strategy
        if cfg.strategy == "naive-rollouts":
            update_behavior_markers = False
            use_cmaes = False
        elif cfg.strategy == "greedy-rollouts":
            update_behavior_markers = False
            use_cmaes = False
        elif cfg.strategy == "behavior-sampling":
            update_behavior_markers = True
            use_cmaes = False
        elif cfg.strategy == "behavior-cmaes":
            update_behavior_markers = True
            use_cmaes = True

    while used_budget < budget:
        # update the acting keys - for stochasticity
        new_acting_keys = cfg.new_acting_keys
        if new_acting_keys:
            random_key, subkey = jax.random.split(random_key)
            acting_keys = get_acting_keys(
                subkey,
                None,
                num_problems,
                cfg.validation_pop_size,
            )
            acting_keys = spread_over_devices(acting_keys, devices=devices)

        if update_behavior_markers:
            if use_cmaes:
                random_key, subkey = jax.random.split(random_key)
                subkeys = jax.random.split(subkey, num=num_problems)
                behavior_markers, _random_keys = jax.vmap(emitter.sample)(
                    emitter_state, subkeys
                )

                noshard_behavior_markers = behavior_markers
            else:
                random_key, subkey = jax.random.split(random_key)
                behavior_markers = cfg.behavior_amplification * jax.random.uniform(
                    subkey,
                    shape=(
                        num_problems,
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
            problems, acting_keys, behavior_markers
        )
        episode_return = jnp.concatenate(episode_return, axis=0)

        if use_cmaes:
            # sort behavior markers based on the perf we got
            fitnesses = -episode_return

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
                                  :, : (cfg.validation_pop_size // 4) * 3
                                  ],
            )

        episode_return = episode_return[:cfg.problems.num_problems]

        latest_batch_best = episode_return.max(-1)
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
        latest_batch_best_sp = episode_return
        latest_batch_min = latest_batch_best_sp.min(-1)
        latest_batch_mean = latest_batch_best_sp.mean(-1)
        latest_batch_std = latest_batch_best_sp.std(-1)

        # Make new metrics dictionary which will be all the returned statistics.
        metrics = {
            f"{return_str}_latest_batch": ret_sign * latest_batch_best.mean(),
            f"{return_str}": ret_sign * best_episode_return.mean(),
            f"{return_str}_latest_min": ret_sign * latest_batch_min.mean(),
            f"{return_str}_latest_min_all": ret_sign * latest_batch_min.min(),
            f"{return_str}_latest_mean": ret_sign * latest_batch_mean.mean(),
            f"{return_str}_latest_std": latest_batch_std.mean(),
        }

        episode_return = episode_return.max(-1)

        incomplete = (episode_return.reshape(
            -1) * ret_sign == 2 * environment.get_episode_horizon()).mean()
        invalid = (episode_return.reshape(
            -1) * ret_sign > 2 * environment.get_episode_horizon()).mean()
        complete = (episode_return.reshape(
            -1) * ret_sign <= environment.get_episode_horizon()).mean()

        metrics[f"incomplete_episodes"] = incomplete
        metrics[f"invalid_episodes"] = invalid
        metrics[f"complete_episodes"] = complete

        # update the used budget
        used_budget += cfg.validation_pop_size
        print("Used budget: ", used_budget)

        log(metrics, used_budget, logger, "slowrl")

    return metrics
