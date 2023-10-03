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

import jax
import jax.numpy as jnp
import numpy as np


def fetch_from_devices(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        if as_numpy and isinstance(x, jax.pxla.ShardedDeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def reduce_from_devices(x, axis=0, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        if isinstance(x, jax.pxla.ShardedDeviceArray):
            if as_numpy:
                x = np.asarray(x)
            x = x.mean(axis=axis)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def fetch_from_first_device(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        x = x[0]
        if as_numpy and isinstance(x, jax.xla.DeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def spread_over_devices(x, devices=None, as_sharded_array=True):
    """Converts a single-device jnp array to a distributed jnp array."""
    devices = devices or jax.local_devices()

    def distribute_fn(x):
        x = x.reshape(len(devices), -1, *(x.shape[1:]))
        x = list(x)
        if as_sharded_array:
            x = jax.device_put_sharded(x, devices)
        return x

    return jax.tree_util.tree_map(distribute_fn, x)


def generate_zeros_from_spec(spec: jnp.ndarray) -> jnp.ndarray:
    zeros: jnp.ndarray = jnp.zeros(spec.shape, spec.dtype)
    return zeros
