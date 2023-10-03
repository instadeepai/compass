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

import hydra
import jax
import omegaconf

from compass.trainers import validate_routing, validate_packing
from compass.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [f"{cfg.algo_name}", "fastrl", f"{cfg.env_name}"]
        neptune_config["parameters"] = cfg

        loggers.append(NeptuneLogger(**neptune_config))

    # return the loggers
    return EnsembleLogger(loggers)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp_tsp",
)
def run(cfg: omegaconf.DictConfig) -> None:
    # Check and configure the available devices.
    behavior_dim = cfg.behavior_dim
    validation_cfg = cfg.validation

    available_devices = len(jax.local_devices())
    if validation_cfg.num_devices < 0:
        validation_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= validation_cfg.num_devices
        ), f"{validation_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    cfg.logger.neptune.name = "fastrl-" + cfg.logger.neptune.name
    logger = create_logger(cfg)

    # init the random key
    random_key = jax.random.PRNGKey(validation_cfg.problem_seed)

    if cfg.env_name == "jssp":
        metrics = validate_packing(
            random_key=random_key,
            cfg=validation_cfg,
            behavior_dim=behavior_dim,
        )
    else:
        metrics = validate_routing(
            random_key=random_key,
            cfg=validation_cfg,
            behavior_dim=behavior_dim,
        )

    metrics = {f"validate/{k}": v for (k, v) in metrics.items()}
    logger.write(metrics)


if __name__ == "__main__":
    run()
