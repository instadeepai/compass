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

from typing import Optional, Sequence, NamedTuple
import math

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from compass.environments.jobshop.environment import JobShop
from compass.environments.jobshop.types import Observation
from jumanji.training.networks.actor_critic import FeedForwardNetwork
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock
from jumanji.training.networks.parametric_distribution import ParametricDistribution


class JobShopParams(NamedTuple):
    critic_encoder: hk.Params
    critic_decoder: hk.Params
    actor_encoder: hk.Params
    actor_decoder: hk.Params


class JobShopNetworks(NamedTuple):
    critic_encoder: FeedForwardNetwork
    critic_decoder: FeedForwardNetwork
    actor_encoder: FeedForwardNetwork
    actor_decoder: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


def make_actor_critic_networks_job_shop(
    job_shop: JobShop,
    num_layers_machines: int,
    num_layers_operations: int,
    num_layers_joint_machines_jobs: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    actor_decoder_mlp_units: Sequence[int],
    critic_decoder_mlp_units: Sequence[int],
) -> JobShopNetworks:
    """Create an actor-critic network for the `JobShop` environment."""
    num_values = np.asarray(job_shop.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    critic_encoder = make_encoder(
        num_layers_machines=num_layers_machines,
        num_layers_operations=num_layers_operations,
        num_layers_joint_machines_jobs=num_layers_joint_machines_jobs,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    critic_decoder = make_critic_decoder(
        decoder_mlp_units=critic_decoder_mlp_units,
    )
    actor_encoder = make_encoder(
        num_layers_machines=num_layers_machines,
        num_layers_operations=num_layers_operations,
        num_layers_joint_machines_jobs=num_layers_joint_machines_jobs,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    actor_decoder = make_actor_decoder(
        decoder_mlp_units=actor_decoder_mlp_units,
    )
    return JobShopNetworks(
        critic_encoder=critic_encoder,
        critic_decoder=critic_decoder,
        actor_encoder=actor_encoder,
        actor_decoder=actor_decoder,
        parametric_action_distribution=parametric_action_distribution,
    )


class JobShopEncoder(hk.Module):
    def __init__(
        self,
        num_layers_machines: int,
        num_layers_operations: int,
        num_layers_joint_machines_jobs: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_layers_machines = num_layers_machines
        self.num_layers_operations = num_layers_operations
        self.num_layers_joint_machines_jobs = num_layers_joint_machines_jobs
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def __call__(self, observation: Observation) -> chex.Array:
        # Machine encoder
        m_remaining_times = observation.machines_remaining_times.astype(float)[
            ..., None
        ]  # (B, M, 1)
        machine_embeddings = self.self_attention_machines(
            m_remaining_times
        )  # (B, M, D)

        # Job encoder
        o_machine_ids = observation.problem[:, 1]  # (B, J, O)
        o_durations = observation.problem[:, 0].astype(float)  # (B, J, O)
        o_mask = observation.ops_mask  # (B, J, O)
        job_embeddings = jax.vmap(
            self.job_encoder, in_axes=(-2, -2, -2, None), out_axes=-2
        )(
            o_durations,
            o_machine_ids,
            o_mask,
            machine_embeddings,
        )  # (B, J, D)
        # Add embedding for no-op
        no_op_emb = hk.Linear(self.model_size)(
            jnp.ones((o_mask.shape[0], 1, 1))
        )  # (B, 1, D)
        job_embeddings = jnp.concatenate(
            [job_embeddings, no_op_emb], axis=-2
        )  # (B, J+1, D)

        # Joint (machines & jobs) self-attention
        embeddings = jnp.concatenate(
            [machine_embeddings, job_embeddings], axis=-2
        )  # (M+J+1, D)
        embeddings = self.self_attention_joint_machines_ops(embeddings)
        return embeddings

    def self_attention_machines(self, m_remaining_times: chex.Array) -> chex.Array:
        # Projection of machines' remaining times
        embeddings = hk.Linear(self.model_size, name="remaining_times_projection")(
            m_remaining_times
        )  # (B, M, D)
        for block_id in range(self.num_layers_machines):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_layers_machines,
                model_size=self.model_size,
                name=f"self_attention_remaining_times_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
            )  # (B, M, D)
        return embeddings

    def job_encoder(
        self,
        o_durations: chex.Array,
        o_machine_ids: chex.Array,
        o_mask: chex.Array,
        m_embedding: chex.Array,
    ) -> chex.Array:
        # Compute mask for self attention between operations
        valid_ops_mask = self._make_self_attention_mask(o_mask)  # (B, 1, O, O)

        # Projection of the operations
        embeddings = hk.Linear(self.model_size, name="durations_projection")(
            o_durations[..., None]
        )  # (B, O, D)

        # Add positional encoding since the operations in each job must be executed sequentially
        max_num_ops = o_durations.shape[-1]
        pos_encoder = PositionalEncoding(
            d_model=self.model_size, max_len=max_num_ops, name="positional_encoding"
        )
        embeddings = pos_encoder(embeddings)

        # Compute cross attention mask
        num_machines = m_embedding.shape[-2]
        cross_attn_mask = self._make_cross_attention_mask(
            o_machine_ids, num_machines, o_mask
        )  # (B, 1, O, M)

        for block_id in range(self.num_layers_operations):
            # Self attention between the operations in the given job
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_layers_operations,
                model_size=self.model_size,
                name=f"self_attention_ops_durations_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=valid_ops_mask
            )

            # Cross attention between the job's ops embedding and machine embedding
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_layers_operations,
                model_size=self.model_size,
                name=f"cross_attention_ops_machines_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=m_embedding,
                value=m_embedding,
                mask=cross_attn_mask,
            )

        embeddings = jnp.sum(embeddings, axis=-2, where=o_mask[..., None])  # (B, D)

        return embeddings

    def self_attention_joint_machines_ops(self, embeddings: chex.Array) -> chex.Array:
        for block_id in range(self.num_layers_joint_machines_jobs):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_layers_joint_machines_jobs,
                model_size=self.model_size,
                name=f"self_attention_joint_ops_machines_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
            )
        return embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)  # (B, O, O)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)  # (B, 1, O, O)
        return mask

    def _make_cross_attention_mask(
        self, o_machine_ids: chex.Array, num_machines: int, o_mask: chex.Array
    ) -> chex.Array:
        # One-hot encode o_machine_ids to satisfy permutation equivariance
        o_machine_ids = jax.nn.one_hot(o_machine_ids, num_machines)  # (B, O, M)
        mask = jnp.logical_and(o_machine_ids, o_mask[..., None])  # (B, O, M)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)  # (B, 1, O, M)
        return mask


def make_encoder(
    num_layers_machines: int,
    num_layers_operations: int,
    num_layers_joint_machines_jobs: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        encoder = JobShopEncoder(
            num_layers_machines=num_layers_machines,
            num_layers_operations=num_layers_operations,
            num_layers_joint_machines_jobs=num_layers_joint_machines_jobs,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="encoder",
        )
        embeddings = encoder(observation)  # (B, M+J+1, D)
        return embeddings

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_decoder(decoder_mlp_units: Sequence[int]) -> FeedForwardNetwork:
    def network_fn(
        embeddings: chex.Array,
        embeddings_st: Optional[chex.Array],
        behavior_marker: chex.Array,
        action_mask: chex.Array,
    ) -> chex.Array:
        """`behavior_marker` is of shape (B, H')."""
        if embeddings_st is not None:
            embeddings = embeddings + embeddings_st  # (B, M+J+1, H)
        behavior_marker = jnp.tile(
            behavior_marker, (1, embeddings.shape[-2], 1)
        )  # (B, M+J+1, H')
        embeddings = jnp.concatenate(
            [embeddings, behavior_marker], axis=-1
        )  # (B, M+J+1, H+H')
        num_machines = action_mask.shape[-2]
        machine_embeddings, job_embeddings = jnp.split(
            embeddings, (num_machines,), axis=-2
        )
        machine_embeddings = hk.nets.MLP(
            decoder_mlp_units, name="actor_decoder_machines"
        )(machine_embeddings)
        job_embeddings = hk.nets.MLP(decoder_mlp_units, name="actor_decoder_jobs")(
            job_embeddings
        )
        logits = jnp.einsum(
            "...mk,...jk->...mj", machine_embeddings, job_embeddings
        )  # (B, M, J+1)

        logits /= jnp.sqrt(decoder_mlp_units[-1])

        logits = jnp.where(action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_decoder(decoder_mlp_units: Sequence[int]) -> FeedForwardNetwork:
    def network_fn(
        embeddings: chex.Array,
        embeddings_st: Optional[chex.Array],
        behavior_marker: chex.Array,
    ) -> chex.Array:
        """`behavior_marker` is of shape (B, H')."""
        if embeddings_st is not None:
            embeddings = embeddings + embeddings_st  # (B, M+J+1, H)
        # Sum embeddings over the sequence length (machines + jobs).
        embedding = jnp.sum(embeddings, axis=-2)  # (B, H)
        embedding = jnp.concatenate([embedding, behavior_marker], axis=-1)  # (B, H+H')
        value = hk.nets.MLP((*decoder_mlp_units, 1), name="critic_decoder")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


class PositionalEncoding(hk.Module):
    def __init__(self, d_model: int, max_len: int, name: Optional[str] = None):
        super(PositionalEncoding, self).__init__(name=name)
        self.d_model = d_model
        self.max_len = max_len

        # Create matrix of shape (max_len, d_model) representing the positional encoding
        # for an input sequence of length max_len
        pos_enc = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=float)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
        pos_enc = pos_enc[None]  # (1, max_len, d_model)
        self.pos_enc = pos_enc

    def __call__(self, embedding: chex.Array) -> chex.Array:
        """Add positional encodings to the embedding for each word in the input sequence.

        Args:
            embedding: input sequence embeddings of shape (B, N, D) where
                B is the batch size, N is input sequence length, and D is
                the embedding dimensionality i.e. d_model.

        Returns:
            Tensor of shape (B, N, D).
        """
        return embedding + self.pos_enc[:, : embedding.shape[1], :]