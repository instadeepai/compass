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

from abc import ABC, abstractmethod
from typing import Union

from chex import Array

from compass.environments.cvrp.types import Observation as CVRPObservation
from compass.environments.tsp.types import Observation as TSPObservation

PoppyObservation = Union[TSPObservation, CVRPObservation]


class PoppyEnv(ABC):
    @abstractmethod
    def get_problem_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_min_start(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_max_start(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_episode_horizon(self) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_problem() -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_observation(*args, **kwargs) -> PoppyObservation:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_reward_negative() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_reward_string() -> str:
        return "return"
