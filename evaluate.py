import argparse

from src.evaluation import Evaluator
from src.utils import ConfigManager, load_yaml

import wandb


def main(config):
    wandb.init(project="exif-sc-attack", config=config, name=config["name"])

    config_manager = ConfigManager(config)

    model = config_manager.init_object("model")
    dataset = config_manager.init_object("dataset")

    evaluator = Evaluator(
        model, dataset, config["adv_step_size"], config["adv_n_iter"], logger=wandb
    )
    results = evaluator(config["save"], tuple(config["resize"]))

    print(results)

    # Flatten nested dict
    log_results = {}
    for type, r in results.items():
        for metric, value in r.items():
            log_results[f"{type}/{metric}"] = value

    wandb.log(log_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to the config file",
        default="configs/evaluate/config.yaml",
    )
    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(config)
