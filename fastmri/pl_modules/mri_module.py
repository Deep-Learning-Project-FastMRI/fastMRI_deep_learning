"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import wandb
import pathlib
import os
import csv
import pickle
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric

import fastmri
from fastmri import evaluate

class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.TestLoss = DistributedMetricSum()
        self.TotTrainExamples = DistributedMetricSum()
        self.TotTrainSliceExamples = DistributedMetricSum()
        self.TotTestExamples = DistributedMetricSum()
        self.TotTestSliceExamples = DistributedMetricSum()
        
         # Initialize metrics tracking dictionaries
        self.epoch_metrics = {
            'epoch': [],
            'nmse': [],
            'ssim': [],
            'psnr': []
        }
        self.best_epoch_metrics = {
            'epoch': 0,
            'nmse': float('inf'),
            'ssim': 0,
            'psnr': 0
        }
        self.train_metrics = {
            'epoch': [],
            'nmse': [],
            'ssim': [],
            'psnr': []
        }
        self.test_metrics = {
            'epoch': [],
            'nmse': [],
            'ssim': [],
            'psnr': []
        }
        self.best_train_metrics = {
            'epoch': 0,
            'nmse': float('inf'),
            'ssim': 0,
            'psnr': 0
        }
        self.best_test_metrics = {
            'epoch': 0,
            'nmse': float('inf'),
            'ssim': 0,
            'psnr': 0
        }

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
           self.val_log_indices = list(range(len(self.trainer.val_dataloaders[0])))


        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

    def test_step_end(self, test_logs):
            # Similar to validation_step_end but for test data
            # check inputs
            for k in (
                "fname",
                "slice",
                "max_value",
                "output",
                "target",
                "test_loss",
            ):
                if k not in test_logs.keys():
                    raise RuntimeError(
                        f"Expected key {k} in dict returned by test_step."
                    )
            # compute evaluation metrics
            mse_vals = defaultdict(dict)
            target_norms = defaultdict(dict)
            ssim_vals = defaultdict(dict)
            max_vals = dict()
            for i, fname in enumerate(test_logs["fname"]):
                slice_num = int(test_logs["slice"][i].cpu())
                maxval = test_logs["max_value"][i].cpu().numpy()
                output = test_logs["output"][i]
                target = test_logs["target"][i]
                
                # print("output shape", output.shape) 
                # print("target shape", target.shape)
                mse_vals[fname][slice_num] = torch.tensor(
                    evaluate.mse(target, output)
                ).view(1)
                target_norms[fname][slice_num] = torch.tensor(
                    evaluate.mse(target, np.zeros_like(target))
                ).view(1)
                ssim_vals[fname][slice_num] = torch.tensor(
                    evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
                ).view(1)
                max_vals[fname] = maxval

            return {
                "test_loss": test_logs["test_loss"],
                "mse_vals": dict(mse_vals),
                "target_norms": dict(target_norms),
                "ssim_vals": dict(ssim_vals),
                "max_vals": max_vals,
                "fname": test_logs["fname"],
                "slice": test_logs["slice"]
            }


    def training_epoch_end(self, outputs):
        # Calculate average loss
        curr_epoch_losses = [x["loss"].detach().cpu().item() for x in outputs]
        avg_loss = np.average(curr_epoch_losses)
        self.log("training_loss", avg_loss, prog_bar=True)
        
        # Print debug information
        wandb.log({
            "epoch": self.current_epoch,
            "train_loss": avg_loss,
        })
        print(f"Train Epoch {self.current_epoch}: Avg Loss = {avg_loss:.4f}")
        
        # We're not calculating detailed metrics for training phase anymore
        # Just save the current metrics for epoch tracking
        current_epoch = self.current_epoch
        self.train_metrics['epoch'].append(current_epoch)
        self.train_metrics['nmse'].append(None)  # No metrics calculated
        self.train_metrics['ssim'].append(None)  # No metrics calculated
        self.train_metrics['psnr'].append(None)  # No metrics calculated

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        wandb.log({
            "epoch": self.current_epoch,
            "val_loss": val_loss / tot_slice_examples,
        })
        # Calculate and log the final metrics for this epoch
        current_epoch = self.current_epoch
        current_metrics = {}
        for metric, value in metrics.items():
            metric_value = value / tot_examples
            current_metrics[metric] = metric_value.cpu().item()
            self.log(f"val_metrics/{metric}", metric_value)
            wandb.log({
                "epoch": self.current_epoch,
                f"val_{metric}": metric_value,
            })
        
        # Store metrics for this epoch
        self.epoch_metrics['epoch'].append(current_epoch)
        self.epoch_metrics['nmse'].append(current_metrics['nmse'])
        self.epoch_metrics['ssim'].append(current_metrics['ssim'])
        self.epoch_metrics['psnr'].append(current_metrics['psnr'])
        
        # Check if this is the best epoch based on SSIM (higher is better)
        if current_metrics['ssim'] > self.best_epoch_metrics['ssim']:
            self.best_epoch_metrics['epoch'] = current_epoch
            self.best_epoch_metrics['nmse'] = current_metrics['nmse']
            self.best_epoch_metrics['ssim'] = current_metrics['ssim']
            self.best_epoch_metrics['psnr'] = current_metrics['psnr']
        
        # print("save metrics being called in validation stuff")
        # Save metrics after each epoch
        self._save_metrics()
        
        # Print debug information
        print(f"Epoch {current_epoch}: SSIM = {current_metrics['ssim']:.4f}, NMSE = {current_metrics['nmse']:.4f}")
        print(f"Best epoch so far: {self.best_epoch_metrics['epoch']} with SSIM = {self.best_epoch_metrics['ssim']:.4f}")


    def test_epoch_end(self, test_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for test_log in test_logs:
            if "test_loss" in test_log:
                losses.append(test_log["test_loss"].view(-1))

            if "mse_vals" in test_log:
                for k in test_log["mse_vals"].keys():
                    mse_vals[k].update(test_log["mse_vals"][k])
                for k in test_log["target_norms"].keys():
                    target_norms[k].update(test_log["target_norms"][k])
                for k in test_log["ssim_vals"].keys():
                    ssim_vals[k].update(test_log["ssim_vals"][k])
                for k in test_log["max_vals"]:
                    max_vals[k] = test_log["max_vals"][k]
        # If we have metrics to calculate
        if mse_vals:
            # check to make sure we have all files in all metrics
            assert (
                mse_vals.keys()
                == target_norms.keys()
                == ssim_vals.keys()
                == max_vals.keys()
            )

            # apply means across image volumes
            metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
            local_examples = 0
            for fname in mse_vals.keys():
                local_examples = local_examples + 1
                mse_val = torch.mean(
                    torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
                )
                target_norm = torch.mean(
                    torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
                )
                metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
                metrics["psnr"] = (
                    metrics["psnr"]
                    + 20
                    * torch.log10(
                        torch.tensor(
                            max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                        )
                    )
                    - 10 * torch.log10(mse_val)
                )
                metrics["ssim"] = metrics["ssim"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
                )

            # reduce across ddp via sum
            metrics["nmse"] = self.NMSE(metrics["nmse"])
            metrics["ssim"] = self.SSIM(metrics["ssim"])
            metrics["psnr"] = self.PSNR(metrics["psnr"])
            tot_examples = self.TotTestExamples(torch.tensor(local_examples))

            if losses:
                test_loss = self.TestLoss(torch.sum(torch.cat(losses)))
                tot_slice_examples = self.TotTestSliceExamples(
                    torch.tensor(len(losses), dtype=torch.float)
                )
                self.log("test_loss", test_loss / tot_slice_examples)
                wandb.log({
                    "epoch": self.current_epoch,
                    "test_loss": test_loss / tot_slice_examples
                })
            # Calculate and log the final metrics for this epoch
            current_epoch = self.current_epoch
            current_metrics = {}
            # print(metrics.items())
            for metric, value in metrics.items():
                metric_value = value / tot_examples
                current_metrics[metric] = metric_value.cpu().item()
                self.log(f"test_metrics/{metric}", metric_value)
                wandb.log({
                "epoch": self.current_epoch,
                f"test_{metric}": metric_value,
                 })
            
            # Store metrics for this epoch
            self.test_metrics['epoch'].append(current_epoch)
            self.test_metrics['nmse'].append(current_metrics['nmse'])
            self.test_metrics['ssim'].append(current_metrics['ssim'])
            self.test_metrics['psnr'].append(current_metrics['psnr'])
            
            # Check if this is the best epoch based on SSIM (higher is better)
            if current_metrics['ssim'] > self.best_test_metrics['ssim']:
                self.best_test_metrics['epoch'] = current_epoch
                self.best_test_metrics['nmse'] = current_metrics['nmse']
                self.best_test_metrics['ssim'] = current_metrics['ssim']
                self.best_test_metrics['psnr'] = current_metrics['psnr']
            
            # print("Saving test metrics")
            # Save metrics after test
            self._save_test_metrics()
            
            # Print debug information
            print(f"Test Results: SSIM = {current_metrics['ssim']:.4f}, NMSE = {current_metrics['nmse']:.4f}")
            print(f"Best test metrics: Epoch {self.best_test_metrics['epoch']} with SSIM = {self.best_test_metrics['ssim']:.4f}")
        
        # print("save metrics being called in epoch")
        # Also save validation metrics to ensure they're not lost
        self._save_metrics()

    def _save_metrics(self):
        """
        Save validation metrics to CSV and pickle files
        """
        print("Saving validation metrics")
        # define paths
        if hasattr(self, "trainer"):
            save_dir = pathlib.Path(self.trainer.default_root_dir) / "metrics"
        else:
            save_dir = pathlib.Path.cwd() / "metrics"
            
        # create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # save metrics to CSV
        csv_path = save_dir / "val_epoch_metrics.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.epoch_metrics['epoch'])):
                writer.writerow({
                    'epoch': self.epoch_metrics['epoch'][i],
                    'nmse': self.epoch_metrics['nmse'][i],
                    'ssim': self.epoch_metrics['ssim'][i],
                    'psnr': self.epoch_metrics['psnr'][i]
                })
        
        # save best epoch metrics to CSV
        best_csv_path = save_dir / "best_val_epoch_metrics.csv"
        with open(best_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(self.best_epoch_metrics)
        
        # save metrics to pickle files
        pickle_path = save_dir / "val_epoch_metrics.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.epoch_metrics, f)
        
        best_pickle_path = save_dir / "best_val_epoch_metrics.pkl"
        with open(best_pickle_path, 'wb') as f:
            pickle.dump(self.best_epoch_metrics, f)

    def _save_train_metrics(self):
        """
        Save training metrics to CSV and pickle files
        """
        print("Saving train metrics")
        # define paths
        if hasattr(self, "trainer"):
            save_dir = pathlib.Path(self.trainer.default_root_dir) / "metrics"
        else:
            save_dir = pathlib.Path.cwd() / "metrics"
            
        # create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # save metrics to CSV
        csv_path = save_dir / "train_epoch_metrics.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.train_metrics['epoch'])):
                writer.writerow({
                    'epoch': self.train_metrics['epoch'][i],
                    'nmse': self.train_metrics['nmse'][i],
                    'ssim': self.train_metrics['ssim'][i],
                    'psnr': self.train_metrics['psnr'][i]
                })
        
        # save metrics to pickle files
        pickle_path = save_dir / "train_epoch_metrics.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.train_metrics, f)
        
        # save best epoch metrics to CSV
        best_csv_path = save_dir / "best_train_epoch_metrics.csv"
        with open(best_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(self.best_train_metrics)
            
        # save best train metrics to pickle
        best_pickle_path = save_dir / "best_train_epoch_metrics.pkl"
        with open(best_pickle_path, 'wb') as f:
            pickle.dump(self.best_train_metrics, f)
            
    def _save_test_metrics(self):
        """
        Save test metrics to CSV and pickle files
        """
        print("Saving test metrics")
        # define paths
        if hasattr(self, "trainer"):
            save_dir = pathlib.Path(self.trainer.default_root_dir) / "metrics"
        else:
            save_dir = pathlib.Path.cwd() / "metrics"
            
        # create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # save metrics to CSV
        csv_path = save_dir / "test_metrics.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.test_metrics['epoch'])):
                writer.writerow({
                    'epoch': self.test_metrics['epoch'][i],
                    'nmse': self.test_metrics['nmse'][i],
                    'ssim': self.test_metrics['ssim'][i],
                    'psnr': self.test_metrics['psnr'][i]
                })
        
        # save best epoch metrics to CSV
        best_csv_path = save_dir / "best_test_metrics.csv"
        with open(best_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(self.best_test_metrics)
        
        # save metrics to pickle files
        pickle_path = save_dir / "test_metrics.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.test_metrics, f)
        
        best_pickle_path = save_dir / "best_test_metrics.pkl"
        with open(best_pickle_path, 'wb') as f:
            pickle.dump(self.best_test_metrics, f)

    

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser