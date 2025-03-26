import os
import pickle


# for every reconstruction file in benchmark and manual

def compare(ground_truth_folder, reconstruction_test_folder):
    # for every file (should be in both folders):
        # get the ground truth of the test image
        # get the reconstructed image
        # get the ROI for both
        # calculate SSIM/PSNR/MSE/etc
    pass



reconstruction_test_folder = "unet_logging/manual/reconstructions/reconstructions/reconstruction_test"

ground_truth_folder = ".../.../brain_data/multicoil_test/"
compare(ground_truth_folder, reconstruction_test_folder)