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

from compass.trainers import TrainerRouting, TrainerPacking
from compass.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [f"{cfg.algo_name}", "training", f"{cfg.env_name}"]
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
    available_devices = len(jax.local_devices())
    if cfg.num_devices < 0:
        cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= cfg.num_devices
        ), f"{cfg.num_devices} devices requested but only {available_devices} available."

    # Create the logger and save the config.
    logger = create_logger(cfg)

    # Train!
    if cfg.env_name == "jssp":
        trainer = TrainerPacking(cfg, logger)
    else:
        trainer = TrainerRouting(cfg, logger)
    trainer.train()

    # Tidy.
    logger.close()


if __name__ == "__main__":
    run()
