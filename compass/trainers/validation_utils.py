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

import os
import pickle

import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf

from compass.utils import (
    spread_over_devices,
    get_acting_keys,
    get_start_positions,
    prepare_problem_batch
)


def get_params(cfg: omegaconf.DictConfig):
    """Load the encoder and decoder parameters from checkpoint.

    Args:
        cfg: The config containing the checkpointing information.

    Returns: The encoder and decoder parameters.
    """
    cfg.checkpoint_fname_load = os.path.splitext(cfg.checkpoint_fname_load)[0]
    if cfg.restore_path:
        with open(
            os.path.join(cfg.restore_path, cfg.checkpoint_fname_load + ".pkl"), "rb"
        ) as f:
            saved_state = pickle.load(f)
        return saved_state.params
    else:
        raise ValueError(
            "Set 'checkpointing.restore_path' in config to the path containing the checkpoint"
        )


def load_instances(cfg, key, environment, num_start_positions, num_agents, batch_size=None):
    """Load problems instances from the given file and generate start positions and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: The PRNGKey for generating the starting positions and acting keys.
        environment: The environment to generate the starting positions on.
        num_start_positions: The number of starting positions to generate.
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    with open(cfg.load_path, "rb") as f:
        problems = jnp.array(pickle.load(f))

    devices = jax.local_devices()
    if len(problems) % len(devices) != 0 and batch_size:
        extra_problems = len(problems) % len(devices)
        while (len(problems) + extra_problems) % (len(devices) * batch_size) != 0:
            extra_problems += len(devices)
        problems = jnp.concatenate([problems, problems[:extra_problems]], axis=0)

    start_key, act_key = random.split(key, 2)
    num_start_positions, start_positions = get_start_positions(
        environment, start_key, num_start_positions, problems.shape[0], num_agents
    )
    acting_keys = get_acting_keys(
        act_key, num_start_positions, problems.shape[0], num_agents
    )

    return problems, start_positions, acting_keys


def get_instances(cfg, key, environment, params, num_start_positions, pop_size, batch_size=None):
    """Get the problem instances, start positions, and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: A PRNGKey.
        environment: The environment to generate the starting positions on.
        params: The encoder and decoder parameters.
        num_start_positions: The number of starting positions to generate.

    Returns:
        problems: A batch of N problems divided over D devices ([D, N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair divided over D devices
        ([D, N, K, M]).
        acting_keys: M acting keys for each problem-agent pair divided over D devices
        ([D, N, K, M, 2]).
    """
    num_agents = pop_size

    if cfg.load_problem:
        problems, start_positions, acting_keys = load_instances(
            cfg, key, environment, num_start_positions, num_agents, batch_size
        )
        problems = spread_over_devices(problems)
        start_positions = spread_over_devices(start_positions)
        acting_keys = spread_over_devices(acting_keys)
    else:
        num_devices = len(jax.local_devices())
        problems, start_positions, acting_keys = jax.vmap(
            prepare_problem_batch, in_axes=(0, None, None, None, None)
        )(
            jax.random.split(key, num_devices),
            environment,
            cfg.num_problems // num_devices,
            num_agents,
            num_start_positions,
        )

    return problems, start_positions, acting_keys