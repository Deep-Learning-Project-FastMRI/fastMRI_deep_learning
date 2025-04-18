import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import h5py  # For reading h5 files

# Add the parent directory to the path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simple test model class with just the methods we need to test
class TestModel:
    def __init__(self, output_path="./test_output"):
        self.output_path = Path(output_path)
    
    def _apply_nms_to_regions(self, binary_mask, min_region_size=10):
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Filter regions based on size (NMS-like behavior)
        refined_mask = np.zeros_like(binary_mask)
        
        # Start from 1 to skip background (labeled as 0)
        for i in range(1, num_labels):
            # Get region size (area)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Keep only regions larger than minimum size
            if area >= min_region_size:
                refined_mask[labels == i] = 1
        
        return refined_mask

    def _detect_hough_lines(self, binary_mask, min_line_length=30, max_line_gap=15):
        # Create a copy of the mask to draw lines on
        line_mask = binary_mask.copy()
        
        # Detect edges using Canny (if not already edges)
        edges = cv2.Canny(binary_mask.astype(np.uint8) * 255, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        # If lines were found, draw them on the mask
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw thicker lines to create ROI regions around detected lines
                cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness=5)
        
        # Combine the original mask with the line mask
        combined_mask = np.maximum(binary_mask, line_mask)
        
        return combined_mask

    def generate_roi_mask_from_input(self, input_image, threshold_ratio=0.05):
        # Normalize the input image to 0-1 range
        if input_image.max() > 0:  # Avoid division by zero
            normalized_image = input_image / input_image.max()
        else:
            normalized_image = input_image.copy()
        
        # Apply gradient magnitude calculation to find edges/features
        sobel_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Apply thresholding to the gradient magnitude
        threshold = threshold_ratio * gradient_magnitude.max()
        binary_mask = (gradient_magnitude > threshold).astype(np.uint8)
        
        # Apply more gentle morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply NMS with smaller minimum region size
        if binary_mask.sum() > 0:  # Only process if we found some regions
            binary_mask = self._apply_nms_to_regions(binary_mask, min_region_size=10)
            binary_mask = self._detect_hough_lines(binary_mask, min_line_length=30, max_line_gap=15)
        
        return binary_mask

def load_mri_from_h5(file_path, slice_idx=10):
    """
    Load an MRI slice from a fastMRI h5 file.
    
    Args:
        file_path (str): Path to the h5 file
        slice_idx (int): Index of the slice to load
        
    Returns:
        np.ndarray: MRI slice as a 2D array
    """
    with h5py.File(file_path, 'r') as hf:
        # Typical fastMRI files have 'reconstruction_esc' or 'reconstruction_rss' keys
        if 'reconstruction_esc' in hf:
            data = hf['reconstruction_esc'][:]
        elif 'reconstruction_rss' in hf:
            data = hf['reconstruction_rss'][:]
        else:
            # If standard keys are not available, try to find image data
            for key in hf.keys():
                if isinstance(hf[key], h5py.Dataset) and len(hf[key].shape) >= 3:
                    data = hf[key][:]
                    break
        
        # Get a specific slice (middle slice if slice_idx is out of bounds)
        if slice_idx >= data.shape[0]:
            slice_idx = data.shape[0] // 2
        
        mri_slice = data[slice_idx]
        
        # Normalize to 0-1 range for visualization
        mri_slice = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min())
        
        return mri_slice

def test_real_mri_roi_detection():
    # Create output directory
    output_dir = Path("./test_real_mri")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Path to the MRI file (update this to the correct path)
    mri_file = "/storage/ice1/3/0/sfarooqui34/fastMRI_deep_learning/b_test/file_brain_AXT2_200_2000167.h5"
    
    # Create a test model instance
    model = TestModel(output_path=str(output_dir))
    
    # Try multiple slices to find good ones
    for slice_idx in [5, 10, 15, 20, 25]:
        try:
            # Load MRI slice
            mri_slice = load_mri_from_h5(mri_file, slice_idx)
            
            # Generate ROI masks with different thresholds
            thresholds = [0.01, 0.03, 0.05, 0.1]
            roi_masks = []
            
            for threshold in thresholds:
                roi_mask = model.generate_roi_mask_from_input(mri_slice, threshold_ratio=threshold)
                roi_masks.append(roi_mask)
            
            # Save individual images
            cv2.imwrite(str(output_dir / f"mri_slice_{slice_idx}.png"), 
                        (mri_slice * 255).astype(np.uint8))
            
            for i, mask in enumerate(roi_masks):
                cv2.imwrite(str(output_dir / f"roi_mask_slice_{slice_idx}_threshold_{thresholds[i]}.png"), 
                            (mask * 255).astype(np.uint8))
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(mri_slice, cmap='gray')
            plt.title(f'MRI Slice {slice_idx}')
            
            plt.subplot(2, 3, 2)
            sobel_x = cv2.Sobel(mri_slice, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(mri_slice, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            plt.imshow(gradient_magnitude, cmap='jet')
            plt.title('Gradient Magnitude')
            
            for i, mask in enumerate(roi_masks):
                plt.subplot(2, 3, i+3)
                plt.imshow(mri_slice, cmap='gray')
                plt.imshow(mask, cmap='hot', alpha=0.3)
                plt.title(f'ROI Overlay (threshold={thresholds[i]})')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / f"roi_visualization_slice_{slice_idx}.png"))
            plt.close()
            
            print(f"Processed slice {slice_idx}")
        except Exception as e:
            print(f"Error processing slice {slice_idx}: {e}")
    
    print(f"Test completed! Check the output in {output_dir.absolute()}")

if __name__ == "__main__":
    test_real_mri_roi_detection()