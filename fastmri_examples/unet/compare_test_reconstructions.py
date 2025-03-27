import os
import pickle
import h5py
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as etree
from fastmri.data.mri_data import FastMRIRawDataSample, et_query
from UnetModuleManual import training_step
from fastmri import evaluate

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from tqdm import tqdm


# for every reconstruction file in benchmark and manual


def compare(ground_truth_folder, reconstruction_test_folder):
    # for every file (should be in both folders):
        # get the ground truth of the test image
        # get the reconstructed image
        # get the ROI for both
        # calculate SSIM/PSNR/MSE/etc
    
    ground_truth_files = sorted(os.listdir(ground_truth_folder))
    # print("num files", len(ground_truth_files)) # 18
    ground_truth_raw_samples = get_raw_gt_samples(ground_truth_files, ground_truth_folder)
    ground_truth_targets = get_targets(ground_truth_raw_samples)
    assert len(ground_truth_targets) == len(ground_truth_raw_samples)
    # print("len ground truth target", len(ground_truth_targets)) # 288
    reconstructions = get_test_reconstructions(reconstruction_test_folder, ground_truth_folder)
    assert len(reconstructions) == len(ground_truth_targets)

    avg_ssim, avg_psnr, avg_nmse, avg_l1_loss = compute_metrics(target_list=ground_truth_targets, reconstruction_list=reconstructions)
    print("avg ssim", avg_ssim)
    print("avg psnr", avg_psnr)
    print("avg nmse", avg_nmse)
    print("avg l1 loss", avg_l1_loss)

    avg_ssim_roi, avg_psnr_roi, avg_nmse_roi, avg_l1_loss_roi = compute_metrics_roi(target_list=ground_truth_targets, reconstruction_list=reconstructions)
    print("avg ssim_roi", avg_ssim_roi)
    print("avg psnr_roi", avg_psnr_roi)
    print("avg nmse_roi", avg_nmse_roi)
    print("avg l1 loss_roi", avg_l1_loss_roi)


def get_test_reconstructions(reconstruction_test_folder, ground_truth_folder):
    test_truth_files = sorted(os.listdir(reconstruction_test_folder))
    reconstruction_list = []
    for fname in test_truth_files:
        metadata, num_slices = retrieve_metadata(fname, ground_truth_folder)

        with h5py.File(reconstruction_test_folder + "/" + fname, "r") as hf:
            reconstruction = hf["reconstruction"]
            curr_reconstruction_list = []
            for slice_ind in range(num_slices):
                curr_reconstruction_list.append(reconstruction[slice_ind, 0, :, :])
            
            reconstruction_list += curr_reconstruction_list
    # print("reconstruction list", len(reconstruction_list))
    return reconstruction_list

def get_raw_gt_samples(ground_truth_files, ground_truth_folder):
    raw_samples = []
    for fname in ground_truth_files:
        metadata, num_slices = retrieve_metadata(fname, ground_truth_folder)
        new_raw_samples = []
        for slice_ind in range(num_slices):
            raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
            new_raw_samples.append(raw_sample)
        raw_samples += new_raw_samples
    return raw_samples

def compute_metrics(target_list, reconstruction_list): 
    # do this function, but only for the ROI (use ananya's function for that)
    ssim_list = []
    pnsr_list = []
    nmse_list = []
    l1_loss_list = []

    for i in range(0, len(target_list), 1):
        target, maxval = target_list[i]
        reconstruction = reconstruction_list[i]
        ssim = torch.tensor(
                evaluate.ssim(target[None, ...], reconstruction[None, ...], maxval=maxval)
            ).view(1)
        psnr = torch.tensor(
            evaluate.psnr(target[None, ...], reconstruction[None, ...], maxval=maxval)
        )
        nmse = torch.tensor(
            evaluate.nmse(roi_target[None, ...], roi_reconstruction[None, ...])
        )
        l1_val = F.l1_loss(roi_target, roi_reconstruction)

        ssim_list.append(ssim)
        pnsr_list.append(psnr)
        nmse_list.append(nmse)
        l1_loss_list.append(l1_val)

    avg_ssim = np.average(ssim_list)
    avg_psnr = np.average(pnsr_list)
    avg_nmse = np.average(nmse_list)
    avg_l1_loss = np.average(l1_loss_list)

    return avg_ssim, avg_psnr, avg_nmse, avg_l1_loss

def compute_metrics_roi(target_list, reconstruction_list): 
    # do this function, but only for the ROI (use ananya's function for that)
    ssim_list = []
    pnsr_list = []
    nmse_list = []
    l1_loss_list = []

    for i in range(0, len(target_list), 1):
        target, maxval = target_list[i]
        reconstruction = reconstruction_list[i]
        H, W = target.shape
        
        center_h, center_w = H // 2, W // 2
        half_size = 200 // 2  

        roi_target = target[center_h - half_size : center_h + half_size,
                            center_w - half_size : center_w + half_size]
        roi_reconstruction = reconstruction[center_h - half_size : center_h + half_size, center_w - half_size : center_w + half_size]
        
        ssim = torch.tensor(
            evaluate.ssim(roi_target[None, ...], roi_reconstruction[None, ...], maxval=maxval)
        ).view(1)
        psnr = torch.tensor(
            evaluate.psnr(roi_target[None, ...], roi_reconstruction[None, ...], maxval=maxval)
        )
        nmse = torch.tensor(
            evaluate.nmse(roi_target[None, ...], roi_reconstruction[None, ...])
        )
        l1_val = F.l1_loss(roi_target, roi_reconstruction)
        
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        nmse_list.append(nmse)
        l1_loss_list.append(l1_val)

    avg_ssim = np.average(ssim_list)
    avg_psnr = np.average(pnsr_list)
    avg_nmse = np.average(nmse_list)
    avg_l1_loss = np.average(l1_loss_list)

    return avg_ssim, avg_psnr, avg_nmse, avg_l1_loss

def get_targets(raw_samples):
    targets = []
    for i in range(0, len(raw_samples), 1):
        # raw_sample = raw_samples[i]
        fname, dataslice, metadata = raw_samples[i]
        # print(metadata.keys())
        with h5py.File(ground_truth_folder + "/" + fname, "r") as hf:
            target = hf["reconstruction_rss"][dataslice]
            targets.append((target, metadata["max"]))
    return targets

def retrieve_metadata(fname, folder):
        with h5py.File(folder + "/" + fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }
        return metadata, num_slices

reconstruction_test_folder = "unet_logging/manual/reconstructions/reconstructions_test/"

# reconstruction_test_folder = None

ground_truth_folder = "../../brain_data/multicoil_test/"
compare(ground_truth_folder, reconstruction_test_folder)