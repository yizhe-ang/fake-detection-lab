import argparse

import pytorch_lightning as pl

from src.models import EXIF_Net
from src.trainers import EXIF_Trainer1, EXIF_Trainer2
from src.utils import ConfigManager, load_yaml

pl.seed_everything(42, workers=True)


def main(config, args):
    # Initialize logger
    if args.wandb:
        logger = pl.loggers.WandbLogger(
            name=config["name"],
            project="exif-sc-train",
        )
        logger.log_hyperparams(config)
    else:
        logger = None

    config_manager = ConfigManager(config)

    net = EXIF_Net(n_attrs=config["datamodule_args"]["n_exif_attr"])

    # Stage 1 Training #########################################################
    dm = config_manager.init_object("datamodule", label="attr")
    dm.prepare_data()
    dm.setup()
    exif_trainer = EXIF_Trainer1(net, dm, config)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{args.checkpoints_dir}/{config['name']}/stage_1",
        monitor="train/loss_1",
        mode="min",
        save_weights_only=True,
    )
    callbacks = [model_checkpoint_callback]

    # Fit model
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        gpus=[args.gpu],
        max_epochs=config["n_epochs_1"],
        deterministic=True,
        benchmark=True,
        # fast_dev_run=True,
    )
    trainer.fit(exif_trainer, datamodule=dm)

    # Stage 2 Training #########################################################
    dm = config_manager.init_object("datamodule", label="img")
    dm.prepare_data()
    dm.setup()
    exif_trainer = EXIF_Trainer2(net, dm, config)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{args.checkpoints_dir}/{config['name']}/stage_2",
        monitor="train/loss_2",
        mode="min",
        save_weights_only=True,
    )
    callbacks = [model_checkpoint_callback]

    # Fit model
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        gpus=[args.gpu],
        max_epochs=config["n_epochs_2"],
        deterministic=True,
        benchmark=True,
        # fast_dev_run=True,
    )
    trainer.fit(exif_trainer, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="path to the config file",
        default="configs/train/exif_sc.yaml",
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="directory to save checkpoint weights",
        default="checkpoints",
    )
    parser.add_argument("--gpu", help="which gpu id to use", type=int, default=0)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to log to Weights & Biases",
    )
    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(config, args)
