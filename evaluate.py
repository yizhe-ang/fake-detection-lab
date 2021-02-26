import argparse

from src.evaluators import Evaluator
from src.utils import ConfigManager, load_yaml


def main(config):
    config_manager = ConfigManager(config)

    model = config_manager.init_object("model")
    dataset = config_manager.init_object("dataset")

    evaluator = Evaluator(model, dataset)
    metrics = evaluator.evaluate(config["save"], tuple(config["resize"]))

    for name, score in metrics.items():
        print(f"{name}: {score}")


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
