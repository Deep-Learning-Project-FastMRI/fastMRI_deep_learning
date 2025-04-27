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
import csv

from .mri_module import MriModule
from collections import defaultdict
import numpy as np
import fastmri
from fastmri import evaluate
import torch.fft as fft
import cv2

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
                output_path="./ROI_generation",
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
        input_image = batch.image.detach().cpu().numpy()[0]
        
        roi_mask = self.generate_roi_mask_from_input(input_image, threshold_ratio=0.7)
        roi_mask_tensor = torch.from_numpy(roi_mask).float().to(batch.image.device)
        
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        self.train_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        
        loss = self.weighted_loss_function(output, batch.target, roi_mask_tensor, roi_weight=2.0)
        self.log("loss", loss.detach())
        
        
        fname = batch.fname[0]
        slice_num = int(batch.slice_num)
        self._save_roi_mask(roi_mask, fname, slice_num, "train")
        
        return loss


    def validation_step(self, batch, batch_idx):
        input_image = batch.image.detach().cpu().numpy()[0]
        roi_mask = self.generate_roi_mask_from_input(input_image)
         
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        self.val_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        
        
        standard_val_loss = F.l1_loss(output, batch.target)
        roi_mask_tensor = torch.from_numpy(roi_mask).float().to(output.device)
        weighted_val_loss = self.weighted_loss_function(output, batch.target, roi_mask_tensor, roi_weight=2.0)
        
        self.log("val_loss", standard_val_loss)
        self.log("weighted_val_loss", weighted_val_loss)

        recon = (output * std + mean).detach().cpu().numpy()[0, ...]
        target = (batch.target * std + mean).detach().cpu().numpy()[0, ...]
        
        error_map = np.abs(recon - target)
        fname = batch.fname[0]
        slice_num = int(batch.slice_num)

        self._original_image(recon, fname, slice_num, "val", "recon")
        self._original_image(target, fname, slice_num, "val", "target")
        self._original_image(input_image, fname, slice_num, "val", "input")
        self._heatmap(error_map, fname, slice_num, split="val")
        
        self._save_roi_mask(roi_mask, fname, slice_num, "val")
        
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": standard_val_loss,
            "weighted_val_loss": weighted_val_loss,
            "roi_mask": roi_mask
        }

    def _save_roi_mask(self, roi_mask, fname, slice_num, split):
        """Save the ROI mask as both an image and a numpy array"""
        mask_dir_img = Path(self.output_path) / "roi_masks" / split / "images"
        mask_dir_npy = Path(self.output_path) / "roi_masks" / split / "numpy"
        
        mask_dir_img.mkdir(parents=True, exist_ok=True)
        mask_dir_npy.mkdir(parents=True, exist_ok=True)
        
        mask_vis = (roi_mask * 255).astype(np.uint8)
        img_file = mask_dir_img / f"{fname}_slice{slice_num:03d}_roi_mask.png"
        cv2.imwrite(str(img_file), mask_vis)
        
        npy_file = mask_dir_npy / f"{fname}_slice{slice_num:03d}_roi_mask.npy"
        np.save(str(npy_file), roi_mask)

    def test_step(self, batch, batch_idx):
        input_image = batch.image.detach().cpu().numpy()[0]
        
        roi_mask = self.generate_roi_mask_from_input(input_image)
        
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        
        self.test_outputs[batch.fname[0]].append((batch.slice_num, output * std + mean))
        
        test_loss = F.l1_loss(output, batch.target)
        self.log("test_loss", test_loss)

        recon = (output * std + mean).detach().cpu().numpy()[0, ...]
        target = (batch.target * std + mean).detach().cpu().numpy()[0, ...]
        error_map = np.abs(recon - target)
        fname = batch.fname[0]
        slice_num = int(batch.slice_num)

        self._original_image(recon, fname, slice_num, "test", "recon")
        self._original_image(target, fname, slice_num, "test", "target")
        self._original_image(input_image, fname, slice_num, "test", "input")
        self._heatmap(error_map, fname, slice_num, split="test")
        
        self._save_roi_mask(roi_mask, fname, slice_num, "test")
        
        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "max_value": batch.max_value,
            "output": (output * std + mean).cpu().numpy(),
            "target": (batch.target * std + mean).cpu().numpy(),
            "test_loss": test_loss,
        }

    def test_step_end(self, test_logs):
        for k in (
            "fname",
            "slice",
            "max_value",
            "output",
            "target",
            "test_loss",
        ):
            if k not in test_logs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by test_step.")
        
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        psnr_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        
        roi_mse_vals = defaultdict(dict)
        roi_ssim_vals = defaultdict(dict)
        roi_psnr_vals = defaultdict(dict)
        roi_nmse_vals = defaultdict(dict)

        for i, fname in enumerate(test_logs["fname"]):
            slice_num = int(test_logs["slice"][i].cpu())
            maxval = test_logs["max_value"][i].cpu().numpy()
            output = test_logs["output"][i]
            target = test_logs["target"][i]
            
            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            psnr_vals[fname][slice_num] = torch.tensor(
                evaluate.psnr(target, output, maxval=maxval)
            ).view(1)
            nmse_vals[fname][slice_num] = torch.tensor(
                evaluate.nmse(target, output)
            ).view(1)

            max_vals[fname] = maxval
            
            roi_mask_path = Path(self.output_path) / "roi_masks" / "test" / "numpy" / f"{fname}_slice{slice_num:03d}_roi_mask.npy"
            if roi_mask_path.exists():
                roi_mask = np.load(str(roi_mask_path))

                mask_bool = roi_mask.astype(bool)
                
                target_roi = target[mask_bool]
                output_roi = output[mask_bool]
                
                if mask_bool.sum() > 0:
                    roi_mse_vals[fname][slice_num] = torch.tensor(
                        evaluate.mse(target_roi, output_roi)
                    ).view(1)
                    
                    roi_ssim_vals[fname][slice_num] = torch.tensor(
                        evaluate.ssim(target[None, ...] * roi_mask, output[None, ...] * roi_mask, maxval=maxval)
                    ).view(1)
                    
                    roi_psnr_vals[fname][slice_num] = torch.tensor(
                        evaluate.psnr(target_roi, output_roi, maxval=maxval)
                    ).view(1)
                    
                    roi_nmse_vals[fname][slice_num] = torch.tensor(
                        evaluate.nmse(target_roi, output_roi)
                    ).view(1)
                else:
                    roi_mse_vals[fname][slice_num] = torch.tensor(float('nan')).view(1)
                    roi_ssim_vals[fname][slice_num] = torch.tensor(float('nan')).view(1)
                    roi_psnr_vals[fname][slice_num] = torch.tensor(float('nan')).view(1)
                    roi_nmse_vals[fname][slice_num] = torch.tensor(float('nan')).view(1)

        return {
            "test_loss": test_logs["test_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
            "fname": test_logs["fname"],
            "psnr_vals": dict(psnr_vals),
            "nmse_vals": dict(nmse_vals),
            "slice": test_logs["slice"],
            "roi_mse_vals": dict(roi_mse_vals),
            "roi_ssim_vals": dict(roi_ssim_vals),
            "roi_psnr_vals": dict(roi_psnr_vals),
            "roi_nmse_vals": dict(roi_nmse_vals)
        }

    def training_epoch_end(self, train_losses):
        super().training_epoch_end(train_losses)
        
        for fname in self.train_outputs:
            self.train_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.train_outputs[fname])
            ])
        
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.train_outputs, self.output_path / "reconstructions_train")
        
        self.train_outputs = defaultdict(list)


    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)

        for fname in self.val_outputs:
            self.val_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.val_outputs[fname])
            ])
        
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.val_outputs, self.output_path / "reconstructions_val")
        
        self.val_outputs = defaultdict(list)
    
    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        
        metrics = {"test/nmse": [], "test/ssim": [], "test/psnr": [], 
                "test/roi_nmse": [], "test/roi_ssim": [], "test/roi_psnr": []}
        
        for log in outputs:
            for metric in ["nmse_vals", "ssim_vals", "psnr_vals", 
                        "roi_nmse_vals", "roi_ssim_vals", "roi_psnr_vals"]:
                if metric not in log:
                    continue
                    
                for fname in log[metric]:
                    for slice_num in log[metric][fname]:
                        val = log[metric][fname][slice_num]
                        
                        if torch.isnan(val).any():
                            continue
                            
                        metrics_key = f"test/{metric.replace('_vals', '')}"
                        metrics[metrics_key].append(val)
        
        mean_metrics = {}
        
        for metric, vals in metrics.items():
            if vals:
                mean_val = torch.mean(torch.cat(vals)).item()
                mean_metrics[metric.replace('test/', '')] = mean_val
                self.log(metric, torch.tensor(mean_val))
        
        if self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                "test/roi_nmse": mean_metrics.get('roi_nmse', 0.0),
                "test/roi_psnr": mean_metrics.get('roi_psnr', 0.0),
                "test/roi_ssim": mean_metrics.get('roi_ssim', 0.0)
            })
        
        metrics_dir = Path(self.output_path) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = metrics_dir / "test_metrics.csv"
        
        file_exists = metrics_file.exists()
        
        with open(metrics_file, 'a' if file_exists else 'w', newline='') as f:
            fieldnames = ['epoch', 'nmse', 'ssim', 'psnr', 'roi_nmse', 'roi_psnr', 'roi_ssim']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'epoch': self.current_epoch,
                'nmse': mean_metrics.get('nmse', 0.0),
                'ssim': mean_metrics.get('ssim', 0.0),
                'psnr': mean_metrics.get('psnr', 0.0),
                'roi_nmse': mean_metrics.get('roi_nmse', 0.0),
                'roi_psnr': mean_metrics.get('roi_psnr', 0.0),
                'roi_ssim': mean_metrics.get('roi_ssim', 0.0)
            })
        
        for fname in self.test_outputs:
            self.test_outputs[fname] = np.stack([
                out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out  
                for _, out in sorted(self.test_outputs[fname])
            ])
        
        if hasattr(self, 'output_path') and self.output_path:
            if not Path(self.output_path).exists():
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
            fastmri.save_reconstructions(self.test_outputs, self.output_path / "reconstructions_test")
        
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

    def _original_image(self,
                    img: np.ndarray,
                    fname: str,
                    slice_num: int,
                    split: str,
                    tag: str):

        img_dir = Path(self.output_path) / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        img_err = cv2.normalize(
            img, None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        out_file = img_dir / f"{fname}_slice{slice_num:03d}_{tag}.png"
        cv2.imwrite(str(out_file), img_err)
    
    def _heatmap(self, error_map: np.ndarray, fname: str, slice_num: int, split: str):

        heatmap_dir = Path(self.output_path) / "heatmaps" / split
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        norm_err = cv2.normalize(
            error_map, None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        heatmap_bgr = cv2.applyColorMap(norm_err, cv2.COLORMAP_JET)
        out_file = heatmap_dir / f"{fname}_slice{slice_num:03d}.png"
        cv2.imwrite(str(out_file), heatmap_bgr)

    def _generate_heatmap_mask(self, error_map, threshold_ratio=0.5):
        """
        Generate a binary mask from the error heatmap by thresholding.
        
        Args:
            error_map (np.ndarray): The error map/heatmap to threshold
            threshold_ratio (float): Value between 0 and 1 that determines the threshold
                                    as a percentage of the maximum value in the heatmap
        
        Returns:
            np.ndarray: Binary mask where 1 indicates ROI regions and 0 is non-ROI
        """
        if error_map.max() > 1.0:
            normalized_map = error_map / error_map.max()
        else:
            normalized_map = error_map.copy()
        
        threshold = threshold_ratio * normalized_map.max()

        binary_mask = (normalized_map > threshold).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        return binary_mask

    def _apply_nms_to_regions(self, binary_mask, min_region_size=50):
        """
        Apply a form of non-maximum suppression to keep only significant regions.
        
        Args:
            binary_mask (np.ndarray): Binary mask from thresholding
            min_region_size (int): Minimum region size to keep
        
        Returns:
            np.ndarray: Refined binary mask after NMS
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        refined_mask = np.zeros_like(binary_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_region_size:
                refined_mask[labels == i] = 1
        
        return refined_mask

    def _detect_hough_lines(self, binary_mask, min_line_length=50, max_line_gap=10):
        """
        Use Hough line transform to detect lines in the binary mask.
        These lines can be used to identify potential ROIs.
        
        Args:
            binary_mask (np.ndarray): Binary mask from thresholding
            min_line_length (int): Minimum line length to detect
            max_line_gap (int): Maximum gap between line segments
        
        Returns:
            np.ndarray: Enhanced binary mask with line regions highlighted
        """
        line_mask = binary_mask.copy()
        
        edges = cv2.Canny(binary_mask.astype(np.uint8) * 255, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness=5)
        
        combined_mask = np.maximum(binary_mask, line_mask)
        
        return combined_mask

    def generate_roi_mask_from_input(self, input_image, threshold_ratio=0.05):
        """
        Generate ROI mask directly from the input image.
        
        Args:
            input_image (np.ndarray): The input MRI image
            threshold_ratio (float): Value between 0 and 1 for thresholding
        
        Returns:
            np.ndarray: Binary mask where 1 indicates ROI regions
        """
        if input_image.max() > 0:
            normalized_image = input_image / input_image.max()
        else:
            normalized_image = input_image.copy()
        
        sobel_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        threshold = threshold_ratio * gradient_magnitude.max()
        binary_mask = (gradient_magnitude > threshold).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        if binary_mask.sum() > 0:
            binary_mask = self._apply_nms_to_regions(binary_mask, min_region_size=10)
            binary_mask = self._detect_hough_lines(binary_mask, min_line_length=30, max_line_gap=15)
        
        return binary_mask

    def weighted_loss_function(self, output, target, roi_mask=None, roi_weight=2.0):
        """
        Apply weighted L1 loss function that gives higher importance to ROI regions.
        
        Args:
            output (torch.Tensor): Model output tensor
            target (torch.Tensor): Target tensor (ground truth)
            roi_mask (torch.Tensor, optional): Binary mask where 1s indicate ROI regions
            roi_weight (float): Weight to apply to errors in ROI regions
        
        Returns:
            torch.Tensor: Weighted loss value
        """
        base_loss = F.l1_loss(output, target, reduction='none')
        
        if roi_mask is None:
            return base_loss.mean()
        
        roi_mask = roi_mask.to(output.device)
        
        weights = torch.ones_like(base_loss)
        weights = weights + (roi_weight - 1.0) * roi_mask
        
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()
