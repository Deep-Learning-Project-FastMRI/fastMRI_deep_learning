import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simplified version of the UnetModuleHeatmap class just for testing
class TestModel:
    def __init__(self, output_path="./test_output"):
        self.output_path = Path(output_path)
    
    def _generate_heatmap_mask(self, error_map, threshold_ratio=0.5):
        """Generate a binary mask from the error heatmap by thresholding."""
        # Normalize the error map to 0-1 range if not already normalized
        if error_map.max() > 1.0:
            normalized_map = error_map / error_map.max()
        else:
            normalized_map = error_map.copy()
        
        # Calculate threshold based on the maximum value
        threshold = threshold_ratio * normalized_map.max()
        
        # Create binary mask by thresholding
        binary_mask = (normalized_map > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask (optional)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        return binary_mask

    def _apply_nms_to_regions(self, binary_mask, min_region_size=50):
        """Apply a form of non-maximum suppression to keep only significant regions."""
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

    def _detect_hough_lines(self, binary_mask, min_line_length=50, max_line_gap=10):
        """Use Hough line transform to detect lines in the binary mask."""
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

    def generate_roi_mask(self, error_map, threshold_ratio=0.5, use_nms=True, use_hough=True):
        """Generate the final ROI mask by combining thresholding, NMS, and Hough transform."""
        # Step 1: Generate initial binary mask through thresholding
        binary_mask = self._generate_heatmap_mask(error_map, threshold_ratio)
        
        # Step 2: Apply NMS to filter small regions (optional)
        if use_nms:
            binary_mask = self._apply_nms_to_regions(binary_mask)
        
        # Step 3: Enhance with Hough line detection (optional)
        if use_hough:
            binary_mask = self._detect_hough_lines(binary_mask)
        
        return binary_mask

def test_roi_mask_generation():
    # Create a test instance
    model = TestModel(output_path="./test_output")
    
    # Create a synthetic error map for testing
    # This creates a simple gradient with higher values in the center
    h, w = 320, 320
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x*x + y*y)
    sigma, mu = 0.5, 0.0
    error_map = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    
    # Add some random noise to make it more realistic
    noise = np.random.normal(0, 0.1, error_map.shape)
    error_map = error_map + noise
    error_map = np.clip(error_map, 0, 1)
    
    # Test each component of the ROI mask generation
    # 1. Basic thresholding
    basic_mask = model._generate_heatmap_mask(error_map, threshold_ratio=0.5)
    
    # 2. With NMS applied
    nms_mask = model._apply_nms_to_regions(basic_mask, min_region_size=50)
    
    # 3. With Hough line detection
    hough_mask = model._detect_hough_lines(nms_mask)
    
    # 4. Combined (final) mask
    final_mask = model.generate_roi_mask(error_map)
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save images for inspection
    cv2.imwrite(str(output_dir / "error_map.png"), (error_map * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "basic_mask.png"), (basic_mask * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "nms_mask.png"), (nms_mask * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "hough_mask.png"), (hough_mask * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "final_mask.png"), (final_mask * 255).astype(np.uint8))
    
    # Plot for visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(error_map, cmap='jet')
    plt.title('Error Map')
    
    plt.subplot(2, 3, 2)
    plt.imshow(basic_mask, cmap='gray')
    plt.title('Basic Threshold Mask')
    
    plt.subplot(2, 3, 3)
    plt.imshow(nms_mask, cmap='gray')
    plt.title('After NMS')
    
    plt.subplot(2, 3, 4)
    plt.imshow(hough_mask, cmap='gray')
    plt.title('After Hough Lines')
    
    plt.subplot(2, 3, 5)
    plt.imshow(final_mask, cmap='gray')
    plt.title('Final ROI Mask')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "roi_mask_visualization.png"))
    plt.close()
    
    print(f"Test completed! Check the output in {output_dir.absolute()}")

if __name__ == "__main__":
    test_roi_mask_generation()