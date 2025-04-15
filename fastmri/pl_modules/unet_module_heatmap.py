"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
import wandb
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from fastmri.models import Unet

from .mri_module import MriModule
from collections import defaultdict
import numpy as np
import fastmri
import torch.fft as fft

class UnetModuleHeatmap(MriModule):
    """
    Unet training module with heatmap.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """
    def __init__(self, 
                in_chans=1,
                out_chans=1,
                chans=32,
                num_pool_layers=4,
                drop_prob=0.0,
                lr=0.001,
                lr_step_size=40,
                lr_gamma=0.1,
                weight_decay=0.0,
                output_path="",
                **kwargs):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.weight_decay = weight_decay
        self.train_outputs = defaultdict(list)
        self.val_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        # Convert output_path to Path object if it's a string
        if isinstance(output_path, str) and output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = output_path

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )
        
    
    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)


    def training_step(self, batch, batch_idx):
        output = self(batch.image) # kspace data is converted to an image in UNetSample
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        # This is from benchmark
        self.train_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        
        # Calculate loss
        loss = F.l1_loss(output, batch.target)
        self.log("loss", loss.detach())
        
        # Return just the loss
        return loss


    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        self.val_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        val_loss = F.l1_loss(output, batch.target)
        self.log("val_loss", val_loss)
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": val_loss,
        }

    def test_step(self, batch, batch_idx):
        # print("HERE")
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        
        # Save to test outputs for reconstruction
        self.test_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        
        # Calculate loss for metrics
        # print("here output", output.shape)
        # print("target here", batch.target.shape)
        # print(batch.target)
        test_loss = F.l1_loss(output, batch.target)
        self.log("test_loss", test_loss)
        
        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "max_value": batch.max_value,
            "output": (output * std + mean).cpu().numpy(),
            "target": (batch.target * std + mean).cpu().numpy(),
            "test_loss": test_loss,
        }

    def training_epoch_end(self, train_losses):
        super().training_epoch_end(train_losses)
        # NOTE: Don't call the parent method since format changed
        # Handle tensor outputs directly
        
        # Save training reconstructions
        for fname in self.train_outputs:
            self.train_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.train_outputs[fname])
            ])
        
        # Save the reconstructions to disk
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.train_outputs, self.output_path / "reconstructions_train")
        
        # Clear the outputs for the next epoch
        self.train_outputs = defaultdict(list)


    def validation_epoch_end(self, outputs):

        # Call the parent class implementation for metric calculation
        super().validation_epoch_end(outputs)

        # Save validation reconstructions
        for fname in self.val_outputs:
            self.val_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.val_outputs[fname])
            ])
        
        # Save the reconstructions to disk
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.val_outputs, self.output_path / "reconstructions_val")
        
        # Clear the outputs for the next epoch
        self.val_outputs = defaultdict(list)
    
    def test_epoch_end(self, outputs):
        # Call the parent class implementation for metric calculation
        super().test_epoch_end(outputs)
            
        # Save test reconstructions
        for fname in self.test_outputs:
            self.test_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.test_outputs[fname])
            ])
        
        # Save the reconstructions to disk
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.test_outputs, self.output_path / "reconstructions_test")
        
        # Clear the outputs for the next epoch
        self.test_outputs = defaultdict(list)


    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser