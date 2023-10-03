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

from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from compass.utils.centroids import compute_cvt_centroids
from compass.utils.cmaes import CMAES, CMAESState


class CMAPoolEmitterState(flax.struct.PyTreeNode):
    """
    Emitter state for the pool of CMA emitters.
    This is for a pool of homogeneous emitters.
    Args:
        current_index: the index of the current emitter state used.
        emitter_states: the batch of emitter states currently used.
    """

    current_index: int
    cmaes_states: CMAESState
    counts: jnp.ndarray


class CMAPoolEmitter:
    def __init__(
        self,
        num_states: int,
        population_size: int,
        num_best: int,
        search_dim: int,
        init_sigma: float,
        delay_eigen_decomposition: bool,
        init_minval: jnp.ndarray,
        init_maxval: jnp.ndarray,
        random_key: jnp.ndarray,
    ):
        """Instantiate a pool of homogeneous emitters.
        Args:
            num_states: the number of emitters to consider. We can use a
                single emitter object and a batched emitter state.
            emitter: the type of emitter for the pool.
        """
        self._num_states = num_states
        self._batch_size = population_size

        self._cmaes = CMAES(
            population_size=population_size,
            num_best=num_best,
            search_dim=search_dim,
            fitness_function=None,
            init_sigma=init_sigma,
            mean_init=None,
            delay_eigen_decomposition=delay_eigen_decomposition,
        )

        # compute centroids
        self._centroids, _random_key = compute_cvt_centroids(
            num_descriptors=search_dim,
            num_init_cvt_samples=10000,
            num_centroids=num_states,
            minval=init_minval,
            maxval=init_maxval,
            random_key=random_key,
        )

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
    ) -> CMAPoolEmitterState:
        """
        Initializes the CMA-MEGA emitter
        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.
        Returns:
            The initial state of the emitter.
        """

        def scan_emitter_init(
            _carry: jnp.ndarray, centroid: jnp.ndarray
        ) -> Tuple[jnp.ndarray, CMAESState]:
            cmaes_state = self._cmaes.init(mean_init=centroid)
            return None, cmaes_state

        # init all the emitter states
        _, cmaes_state = jax.lax.scan(
            scan_emitter_init, None, (self._centroids), length=self._num_states
        )

        # define the emitter state of the pool
        emitter_state = CMAPoolEmitterState(
            current_index=0,
            cmaes_states=cmaes_state,
            counts=jnp.zeros(self._num_states),
        )

        return emitter_state

    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        emitter_state: CMAPoolEmitterState,
        random_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # retrieve the relevant emitter state
        current_index = emitter_state.current_index
        used_cmaes_state = jax.tree_util.tree_map(
            lambda x: x[current_index], emitter_state.cmaes_states
        )

        # use it to emit offsprings
        offsprings, random_key = self._cmaes.sample(used_cmaes_state, random_key)

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def update_state(
        self,
        emitter_state: CMAPoolEmitterState,
        sorted_candidates: jnp.ndarray,
    ) -> CMAPoolEmitterState:
        # retrieve the emitter that has been used and it's emitter state
        current_index = emitter_state.current_index
        cmaes_states = emitter_state.cmaes_states
        emit_counts = emitter_state.counts

        used_cmaes_state = jax.tree_util.tree_map(
            lambda x: x[current_index], cmaes_states
        )

        # update the used emitter state
        used_emitter_state = self._cmaes.update_state(
            cmaes_state=used_cmaes_state,
            sorted_candidates=sorted_candidates,
        )

        # update the emitter state
        cmaes_states = jax.tree_util.tree_map(
            lambda x, y: x.at[current_index].set(y), cmaes_states, used_emitter_state
        )

        # update emit counts
        emit_counts = emit_counts.at[current_index].add(1)

        # determine the next emitter to be used
        new_index = jnp.argmin(emit_counts)

        emitter_state = emitter_state.replace(
            current_index=new_index, cmaes_states=cmaes_states, counts=emit_counts
        )

        return emitter_state  # type: ignore
