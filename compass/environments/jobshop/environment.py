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

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs

from jumanji.environments.packing.job_shop.env import JobShop
from compass.environments.jobshop.types import Observation, State

from jumanji.types import termination, transition, restart

from compass.environments.poppy_env import PoppyEnv
from compass.environments.jobshop.utils import generate_problem


class PoppyJobShop(JobShop, PoppyEnv):

    def __init__(self, num_jobs=10, num_machines=10, max_op_duration=99, episode_horizon=1250):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.min_ops_duration = 1
        self.max_op_duration = max_op_duration
        self.max_num_ops = num_machines
        self.no_op_idx = num_jobs

        self.durations = None
        self.episode_horizon = episode_horizon

    def get_episode_horizon(self) -> int:
        return self.episode_horizon

    def get_state(self, problem):
        durations, machines = problem[0], problem[1]

        ops_mask = jnp.ones((self.num_jobs, self.max_num_ops), dtype=bool)

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)

        # Initially, none of the operations have been scheduled
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)

        # Time starts at 0
        step_count = jnp.array(0, jnp.int32)

        state = State(
            ops_machine_ids=machines,
            ops_durations=durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=None,
            step_count=step_count,
            scheduled_times=scheduled_times,
            key=None,
            is_done=jnp.int32(0),
        )

        self.durations = durations
        self.episode_horizon = self.get_episode_horizon()

        return state

    def reset_from_state(self, problem):
        # Generate a random problem
        state = self.get_state(problem)

        # Create the action mask and update the state
        state.action_mask = self._create_action_mask(
            state.machines_job_ids,
            state.machines_remaining_times,
            state.ops_machine_ids,
            state.ops_mask,
        )

        # Get the observation and the timestep
        obs = Observation(
            problem=problem,
            ops_mask=state.ops_mask,
            machines_job_ids=state.machines_job_ids,
            machines_remaining_times=state.machines_remaining_times,
            action_mask=state.action_mask,
            is_done=jnp.int32(0),
        )
        timestep = restart(observation=obs)

        return state, timestep

    def compute_reward(self, state):
        # Check if the schedule has finished
        all_operations_scheduled = ~jnp.any(state.ops_mask)
        schedule_finished = all_operations_scheduled & jnp.all(
            state.machines_remaining_times == 0
        )
        # calculate when the schedule finished
        finished_step_count = (state.scheduled_times + state.ops_durations).max()
        # calculate the reward
        reward = jnp.where(schedule_finished,
                           jnp.array(-finished_step_count, float),
                           jnp.array(-self.num_jobs * self.max_num_ops * self.max_op_duration,
                                     float),
                           )
        return reward

    def step(self, state, action):

        # Conditions for is_done:
        # (1) all operations have been scheduled
        # (2) all machines are idle
        # (3) the episode horizon has been reached
        # (4) the action is invalid

        # Check the action is legal
        invalid = ~jnp.all(state.action_mask[jnp.arange(self.num_machines), action])  # type: ignore

        # Obtain the id for every job's next operation
        op_ids = jnp.argmax(state.ops_mask, axis=-1)

        # Update the status of all machines
        (
            updated_machines_job_ids,
            updated_machines_remaining_times,
        ) = self._update_machines(
            action,
            op_ids,
            state.machines_job_ids,
            state.machines_remaining_times,
            state.ops_durations,
        )

        # Update the status of operations that have been scheduled
        updated_ops_mask, updated_scheduled_times = self._update_operations(
            action,
            op_ids,
            state.step_count,
            state.scheduled_times,
            state.ops_mask,
        )

        # Update the action_mask
        updated_action_mask = self._create_action_mask(
            updated_machines_job_ids,
            updated_machines_remaining_times,
            state.ops_machine_ids,
            updated_ops_mask,
        )

        # Increment the time step
        updated_step_count = jnp.array(state.step_count + 1, jnp.int32)

        # Check if all machines are idle simultaneously
        all_machines_idle = jnp.all(
            (updated_machines_job_ids == self.no_op_idx)
            & (updated_machines_remaining_times == 0)
        )

        # Check if the schedule has finished and horizon is reached
        all_operations_scheduled = ~jnp.any(updated_ops_mask)
        schedule_finished = all_operations_scheduled & jnp.all(
            updated_machines_remaining_times == 0
        )
        horizon_reached = updated_step_count >= self.episode_horizon
        # Compute terminal condition
        done = invalid | all_machines_idle | schedule_finished | horizon_reached | state.is_done.astype(bool)

        # Update the state and extract the next observation
        next_state = State(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            ops_mask=updated_ops_mask,
            machines_job_ids=updated_machines_job_ids,
            machines_remaining_times=updated_machines_remaining_times,
            action_mask=updated_action_mask,
            step_count=updated_step_count,
            scheduled_times=updated_scheduled_times,
            key=state.key,
            is_done=done.astype(int),
        )

        problem = jnp.concatenate((next_state.ops_durations.reshape(1, self.num_jobs, self.num_machines),
                                  next_state.ops_machine_ids.reshape(1, self.num_jobs, self.num_machines)))
        next_obs = Observation(
            problem=problem,
            ops_mask=next_state.ops_mask,
            machines_job_ids=next_state.machines_job_ids,
            machines_remaining_times=next_state.machines_remaining_times,
            action_mask=next_state.action_mask,
            is_done=done.astype(jnp.int32),
        )

        reward = jax.lax.cond(
            state.is_done.astype(bool),
            lambda _: jnp.array(0, float),
            lambda _: jax.lax.cond(invalid | all_machines_idle,
                                   lambda _: jnp.array(-self.num_jobs * self.max_num_ops * self.max_op_duration, float),
                                   # lambda _: jnp.array(-1, float), None),
                                   # if you do not finish the schedule, reward is 2 * episode_horizon
                                   lambda _: jax.lax.cond(~schedule_finished & horizon_reached,
                                                          lambda _: jnp.array(-self.episode_horizon - 1, float),
                                                          lambda _: jnp.array(-1, float), None),
                                   None),
            None,
        )

        timestep = jax.lax.cond(
            horizon_reached,
            termination,
            transition,
            reward,
            next_obs,
        )

        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `JobShop` environment.
                Returns:
                    Spec containing the specifications for all the `Observation` fields:
                    - ops_machine_ids: BoundedArray (int32) of shape (num_jobs, max_num_ops).
                    - ops_durations: BoundedArray (int32) of shape (num_jobs, max_num_ops).
                    - ops_mask: BoundedArray (bool) of shape (num_jobs, max_num_ops).
                    - machines_job_ids: BoundedArray (int32) of shape (num_machines,).
                    - machines_remaining_times: BoundedArray (int32) of shape (num_machines,).
                    - action_mask: BoundedArray (bool) of shape (num_machines, num_jobs + 1).
                """
        problem = specs.BoundedArray(
            shape=(2, self.num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.max_op_duration,
            name="problem",
        )
        ops_mask = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="ops_mask",
        )
        machines_job_ids = specs.BoundedArray(
            shape=(self.num_machines,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_jobs,
            name="machines_job_ids",
        )
        machines_remaining_times = specs.BoundedArray(
            shape=(self.num_machines,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.max_op_duration,
            name="machines_remaining_times",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_machines, self.num_jobs + 1),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            constructor=Observation,
            name="ObservationSpec",
            problem=problem,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=action_mask,
            is_done=specs.DiscreteArray(2, dtype=jnp.int32, name="is_done"),
        )

    def render(self, state: State) -> Any:
        pass

    def get_problem_size(self) -> Tuple[int, int, int]:
        return self.num_machines, self.num_jobs, self.max_op_duration

    def get_min_start(self) -> int:
        return -1

    def get_max_start(self) -> int:
        return -1

    @staticmethod
    def generate_problem(*args, **kwargs) -> chex.Array:
        return generate_problem(*args, **kwargs)

    @staticmethod
    def make_observation(*args, **kwargs) -> Observation:
        return Observation(*args, **kwargs)

    @staticmethod
    def is_reward_negative() -> bool:
        return True

    @staticmethod
    def get_reward_string() -> str:
        return "schedule_duration"
