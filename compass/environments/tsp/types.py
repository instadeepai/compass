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
import chex
import jax.numpy as jnp


class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates for all cities.
    position: index of current city.
    trajectory: array of city indices defining the route (-1 --> not filled yet).
    action_mask: binary mask (False/True <--> illegal/legal).
    """

    problem: chex.Array  # (num_cities, 2)
    start_position: jnp.int32
    position: chex.Numeric  # ()
    trajectory: chex.Array  # (num_cities,)
    action_mask: chex.Array  # (num_cities,)
    is_done: jnp.int32
