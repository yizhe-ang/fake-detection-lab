import argparse
import json

from src.evaluation import NonAdvEvaluator
from src.utils import ConfigManager, load_yaml


def main(config, args):
    config_manager = ConfigManager(config)

    model = config_manager.init_object("model")
    dataset = config_manager.init_object("dataset")

    # Run evaluation
    evaluator = NonAdvEvaluator(model, dataset)
    results = evaluator.evaluate(tuple(config["resize"]))

    # Save results
    print(results)
    with open(args.results_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to the config file",
        default="configs/evaluate/config.yaml",
    )
    parser.add_argument(
        "--results_path",
        help="path to store evaluation results as JSON file",
        default="results.json",
    )
    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(config, args)
