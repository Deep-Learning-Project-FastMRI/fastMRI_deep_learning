import os
import pickle
import h5py
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as etree
from fastmri.data.mri_data import FastMRIRawDataSample, et_query

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


def compare(ground_truth_folder, reconstruction_test_folder):
    # for every file (should be in both folders):
        # get the ground truth of the test image
        # get the reconstructed image
        # get the ROI for both
        # calculate SSIM/PSNR/MSE/etc
    
    ground_truth_files = sorted(os.listdir(ground_truth_folder))
    result = get_raw_gt_samples(ground_truth_files, ground_truth_folder)
    print(len(result))

def get_raw_gt_samples(ground_truth_files, ground_truth_folder):
    raw_samples = []
    for fname in ground_truth_files:
        print(fname)
        metadata, num_slices = retrieve_metadata(fname, ground_truth_folder)

        new_raw_samples = []
        for slice_ind in range(num_slices):
            raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
            new_raw_samples.append(raw_sample)

        raw_samples += new_raw_samples
    return raw_samples


def get_gt(fname):
        print("FNAME HERE IS: ")

        print(fname)
        with h5py.File(ground_truth_folder + "/" + fname, "r") as hf:
            fname, dataslice, metadata = self.raw_samples[i]
            target = hf["reconstruction_rss"][dataslice] 

        return target

# reconstruction_test_folder = "unet_logging/manual/reconstructions/reconstructions/reconstruction_test"

reconstruction_test_folder = None

ground_truth_folder = "../../brain_data/multicoil_test/"
compare(ground_truth_folder, reconstruction_test_folder)