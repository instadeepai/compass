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
from typing import Optional

import chex
import hydra
import jax
import jax.numpy as jnp
import omegaconf

import compass.trainers.packing.trainer_packing as trainer
from compass.utils.metrics import get_metrics
from compass.trainers.validation_utils import get_params, get_instances
from compass.networks.jobshop import make_actor_critic_networks_job_shop


def validate(
        random_key,
        cfg: omegaconf.DictConfig,
        params: chex.ArrayTree = None,
        behavior_dim: Optional[int] = None,
) -> dict:
    """Run validation on input problems.

    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.

    Returns:
        metrics: A dictionary of metrics from the validation.
    """

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, acting_keys, behavior_markers):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
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
        num_agents = acting_keys.shape[1]
        num_batches = int(round(len(problems) / cfg.batch_size, 0))
        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)

        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))
        num_problems = problems.shape[1]

        print("Behavior markers: ", behavior_markers.shape)

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)

            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )
            # Note, the starting positions and acting keys are duplicated here.
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)

        def body(_, x):
            problems, acting_keys = x
            # acting_keys = acting_keys.reshape(acting_keys.shape[0], acting_keys.shape[1], 1, -1)
            _, (traj, info) = trainer.rollout(
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
            body, init=None, xs=(problems, acting_keys)
        )

        if cfg.use_augmentations:
            metrics = jax.tree_map(
                lambda x: x.reshape(
                    num_batches,
                    num_problems,
                    8,
                    num_agents,
                    -1,
                ).max(2),
                metrics,
            )

        # Flatten batch dimension of metrics.
        metrics = jax.tree_map(lambda x: x.reshape(*(-1,) + x.shape[2:]), metrics)
        episode_return = metrics["rewards"].sum(-1)  # [N, K]

        return episode_return

    environment = hydra.utils.instantiate(cfg.environment)
    networks = make_actor_critic_networks_job_shop(job_shop=environment,
                                                   num_layers_machines=cfg.networks.num_layers_machines,
                                                   num_layers_operations=cfg.networks.num_layers_operations,
                                                   num_layers_joint_machines_jobs=cfg.networks.num_layers_joint_machines_jobs,
                                                   transformer_num_heads=cfg.networks.transformer_num_heads,
                                                   transformer_key_size=cfg.networks.transformer_key_size,
                                                   transformer_mlp_units=cfg.networks.transformer_mlp_units,
                                                   actor_decoder_mlp_units=cfg.networks.actor_decoder_mlp_units,
                                                   critic_decoder_mlp_units=cfg.networks.critic_decoder_mlp_units,
                                                   )
    if not params:
        params = get_params(cfg.checkpointing)

    key = jax.random.PRNGKey(cfg.problem_seed)
    problems, _, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
        pop_size=cfg.validation_sample_size,
        batch_size=cfg.batch_size,
    )

    behavior_markers = cfg.behavior_amplification * jax.random.uniform(
        random_key,
        shape=(cfg.validation_sample_size, behavior_dim),
        minval=-1,
        maxval=1,
    )


    # replicate them over the devices
    devices = jax.local_devices()
    behavior_markers = jax.device_put_replicated(behavior_markers, devices)

    episode_return = run_validate(problems, acting_keys, behavior_markers)

    episode_return = jnp.concatenate(episode_return, axis=0)
    episode_return = episode_return[:cfg.problems.num_problems]

    if environment.is_reward_negative():
        ret_sign = -1
    else:
        ret_sign = 1
    return_str = environment.get_reward_string()

    # Make new metrics dictionary which will be all the returned statistics.
    metrics = {
        f"{return_str}": ret_sign * episode_return.max(-1).mean(),
        f"{return_str}_rand_agent": ret_sign * episode_return.mean(),
    }

    metrics = get_metrics(
        metrics,
        episode_return.reshape(episode_return.shape + (1,)),
        compute_expensive_metrics=cfg.compute_expensive_metrics,
    )

    return metrics
