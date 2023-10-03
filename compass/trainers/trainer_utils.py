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

from dataclasses import field
from typing import TYPE_CHECKING, Optional, Union, List

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

import haiku as hk
import jax.numpy as jnp
import jmp
import optax
from chex import Array, PRNGKey
from jumanji.environments.routing.cvrp.types import State as StateCVRP
from jumanji.environments.routing.tsp.types import State as StateTSP
from jumanji.types import TimeStep

from compass.environments.tsp import Observation as ObservationTSP
from compass.environments.cvrp import Observation as ObservationCVRP
from compass.networks.jobshop import JobShopParams



State = Union[StateTSP, StateCVRP]
Observation = Union[ObservationTSP, ObservationCVRP]


@dataclass
class TrainingStatePacking:  # type: ignore
    """Container for data used during training."""

    params: JobShopParams
    behavior_markers: chex.Array
    optimizer_state: optax.OptState
    num_steps: jnp.int32
    key: PRNGKey
    extras: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingStateRouting:  # type: ignore
    """Container for data used during training."""

    params: hk.Params
    decoder_params: hk.Params
    behavior_markers: chex.Array
    optimizer_state: optax.OptState
    num_steps: jnp.int32
    key: PRNGKey
    extras: Optional[dict] = field(default_factory=dict)


@dataclass
class ActingState:  # type: ignore
    """Container for data used during the acting in the environment."""

    state: State
    timestep: TimeStep
    key: PRNGKey


@dataclass
class Information:  # type: ignore
    extras: Optional[dict] = field(default_factory=dict)
    metrics: Optional[dict] = field(default_factory=dict)
    logging: Optional[dict] = field(default_factory=dict)


def get_policy(use_half=True, is_norm_layer=False):
    """Get a jmp.Policy.

    Related to mixed precision training in Jax.

    Args:
        use_half: Whether we are configured to use half (bf16) precision.
        is_norm_layer: Whether this policy should be that for a normalisation layer.

    Returns: A configured jmp.Policy.
    """

    half = jnp.bfloat16  # only support A100 GPU and TPU for now
    full = jnp.float32
    if use_half:
        if is_norm_layer:
            # Compute normalisation layers (batch/layer etc) in full precision to avoid instabilities.
            policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        else:
            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    else:
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)
    return policy


def set_policy(modules: Union[List[hk.Module], hk.Module], use_half: bool = True):
    """Set the jmp.Policy of the passed modules.

    Args:
        modules: A list of (or single) haiku modules.
        use_half: Whether we are configured to use half (bf16) precision.

    Returns: None
    """
    if type(modules) is not list:
        modules = [modules]
    for mod in modules:
        policy = get_policy(use_half, is_norm_layer=False)
        hk.mixed_precision.set_policy(mod, policy)
    if use_half:
        # Ensure some layers are always in full precision.
        policy_norm = get_policy(use_half=True, is_norm_layer=True)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy_norm)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy_norm)