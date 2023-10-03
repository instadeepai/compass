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

from typing import Union

import haiku as hk
import jax.numpy as jnp
from chex import Array

from compass.environments.cvrp import Observation as CVRPObservation
from compass.environments.tsp import Observation as TSPObservation
from compass.environments.cvrp.utils import DEPOT_IDX
from compass.networks.base import DecoderBase, EncoderBase


class CVRPEncoder(EncoderBase):
    # Modified for CVRP according to original source code: https://github.com/yd-kwon/POMO/blob/master/NEW_py_ver/CVRP/POMO/CVRPModel.py (~line 116)
    def get_problem_projection(self, problem: Array) -> Array:
        proj_depot = hk.Linear(self.model_size, name="depot_encoder")
        proj_nodes = hk.Linear(self.model_size, name="nodes_encoder")
        return jnp.where(
            jnp.zeros((problem.shape[0], 1)).at[DEPOT_IDX].set(1),
            proj_depot(problem),
            proj_nodes(problem),
        )


class CVRPDecoder(DecoderBase):
    def get_context(self, observation: CVRPObservation, embeddings: Array) -> Array:  # type: ignore[override]
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                observation.capacity[None],
            ],
            axis=0,
        )[
            None
        ]  # [1, 2*128+1=257,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return jnp.where(attention_mask, 0, 1)


class CVRPConditionedDecoder(CVRPDecoder):
    def __init__(
        self,
        num_heads,
        key_size,
        model_size=128,
        name="decoder",
        embedding_size=16,
    ):
        super().__init__(num_heads, key_size, model_size, name)

        self.embedding_size = embedding_size

    def __call__(
        self,
        observation: Union[TSPObservation, CVRPObservation],
        embeddings: Array,
        behavior_marker: Array,
        condition_query: bool = False,
        condition_key: bool = False,
        condition_value: bool = False,
    ) -> Array:
        context = self.get_context(observation, embeddings)  # [1, 3*128=384,]

        if condition_query:
            context = jnp.concatenate(
                [context, jnp.expand_dims(behavior_marker, axis=0)], axis=1
            )  # with bd: [1, 384 + bd_dim]

        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init_scale=1,
            name="mha_dec",
        )

        mha_value = embeddings
        if condition_value:
            repeated_behavior_marker = jnp.repeat(
                jnp.expand_dims(behavior_marker, axis=0),
                repeats=embeddings.shape[0],
                axis=0,
            )

            mha_value = jnp.concatenate([embeddings, repeated_behavior_marker], axis=1)

        mha_key = embeddings
        if condition_key:
            mha_key = mha_value

        attention_mask = jnp.expand_dims(observation.action_mask, (0, 1))
        context = mha(
            query=context,
            key=mha_key,
            value=mha_value,
            mask=self.get_transformed_attention_mask(attention_mask),
        )  # --> [128]

        attn_logits = (
            embeddings @ context.squeeze() / jnp.sqrt(self.model_size)
        )  # --> [num_cities/items]
        attn_logits = 10 * jnp.tanh(attn_logits)  # clip to [-10,10]

        return attn_logits