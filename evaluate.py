import argparse
import json

import wandb
from src.evaluation import Evaluator
from src.utils import ConfigManager, load_yaml


def main(config, args):
    # Initialize logger
    if args.wandb:
        wandb.init(project="exif-sc-attack", config=config, name=config["name"])
        logger = wandb
    else:
        logger = None

    config_manager = ConfigManager(config)

    model = config_manager.init_object("model")
    dataset = config_manager.init_object("dataset")

    evaluator = Evaluator(
        model,
        dataset,
        adv_step_size=config["adv_step_size"],
        adv_n_iter=config["adv_n_iter"],
        vis_dir=args.vis_dir,
        logger=logger,
    )

    # Run evaluation
    results = evaluator(tuple(config["resize"]))

    # Save results
    print(results)
    with open(args.results_path, "w") as f:
        json.dump(results, f)

    # Log results
    if args.wandb:
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
    parser.add_argument(
        "--results_path",
        help="path to store evaluation results as JSON file",
        default="results.json",
    )
    parser.add_argument(
        "--vis_dir",
        help="directory to save visualization results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to log to Weights & Biases",
    )
    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(config, args)
