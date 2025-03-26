import os
import pickle
import h5py
import numpy as np

# for every reconstruction file in benchmark and manual

def compare(ground_truth_folder, reconstruction_test_folder):
    # for every file (should be in both folders):
        # get the ground truth of the test image
        # get the reconstructed image
        # get the ROI for both
        # calculate SSIM/PSNR/MSE/etc
    
    ground_truth_files = sorted(os.listdir(ground_truth_folder))
    for gt_file in ground_truth_files:
        print(get_gt(gt_file))

def get_gt(fname):
        print("FNAME HERE IS: ")
        print(fname)
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf["reconstruction_rss"][dataslice] 

        return target

# reconstruction_test_folder = "unet_logging/manual/reconstructions/reconstructions/reconstruction_test"

reconstruction_test_folder = None

ground_truth_folder = "../../brain_data/multicoil_test/"
compare(ground_truth_folder, reconstruction_test_folder)