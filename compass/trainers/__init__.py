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

from compass.trainers.routing.trainer_routing import Trainer as TrainerRouting
from compass.trainers.packing.trainer_packing import Trainer as TrainerPacking

from compass.trainers.routing.validation_routing import validate as validate_routing
from compass.trainers.packing.validation_packing import validate as validate_packing

from compass.trainers.routing.slowrl_validation_routing import slowrl_validate as slowrl_validate_routing
from compass.trainers.packing.slowrl_validation_packing import slowrl_validate as slowrl_validate_packing