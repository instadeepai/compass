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

from typing import NamedTuple
import jax.numpy as jnp
from chex import Array


class Observation(NamedTuple):
    """
    problem: array with the coordinates of all nodes (+ depot) and their cost
    position: index of the current node
    capacity: current capacity of the vehicle
    invalid_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array  # (num_nodes + 1, 3)
    position: jnp.int32
    capacity: jnp.float32
    action_mask: Array  # (num_nodes + 1,)
    is_done: jnp.int32
