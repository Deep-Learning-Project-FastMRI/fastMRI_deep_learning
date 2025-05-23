"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.models import Unet
import wandb
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule, UnetModuleManual, UnetModuleHeatmap, UnetAttentionModule


def cli_main(args):
    wandb.login(key="926b1c7d6af6fe4e896235f7787591e9adb48d1e")
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)
    # ptl data module - this handles data loaders
    print(args.data_path)
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dataset_cache_file=False,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------

    if args.experiment_mode == "benchmark":
        model = UnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            output_path = pathlib.Path("unet_logging/benchmark/reconstructions")
        )
    elif (args.experiment_mode == "manual"):
        model = UnetModuleManual(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            output_path = pathlib.Path("unet_logging/manual/reconstructions")
        )
    elif args.experiment_mode == "heatmap":
        model = UnetModuleHeatmap(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        output_path=pathlib.Path("unet_logging/heatmap/reconstructions")
        )
    elif args.experiment_mode == "attention":
        model = UnetAttentionModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        output_path=pathlib.Path("unet_logging/attention/reconstructions")
        )    
    else:
        raise ValueError("Invalid experiment mode")
    
    project_name = "deep_learning_fastmri_project"
    config = {"experiment_mode": args.experiment_mode, "mode":args.mode}
    run_name=f'{args.experiment_mode}' + '_' + f'{args.mode}'
    wandb.init(
        project=project_name,
        config=config
    )
    
    print("Train dataset size", len(data_module.train_dataloader().dataset))
    print("Val dataset size", len(data_module.val_dataloader().dataset))
    print("Test dataset size", len(data_module.test_dataloader().dataset))

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    # ------------
    # run
    # ------------
    torch.use_deterministic_algorithms(True, warn_only=True)

    with wandb.init(config=config) as run:
        run.name=run_name
        if args.mode == "train":
            trainer.fit(model, datamodule=data_module)
        elif args.mode == "test":
            trainer.test(model, datamodule=data_module, ckpt_path = args.resume_from_checkpoint)
        else:
            raise ValueError(f"unrecognized mode {args.mode}")
    
    

def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    num_gpus = 1
    backend = None if num_gpus == 1 else "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    parser.add_argument(
        "--experiment_mode",
        default="benchmark",
        choices=("benchmark", "manual", "heatmap", "attention"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )


    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)
    print(parser)
    # module config
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        # default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
        log_every_n_steps=1,
    )



    args = parser.parse_args()
    default_root_dir = fetch_dir("log_path", path_config) / "unet_logging" / args.experiment_mode

    print(args)
    args.default_root_dir = default_root_dir

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="val_metrics/ssim",
            mode="max",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
