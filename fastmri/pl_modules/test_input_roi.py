import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the parent directory to the path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simple test model class with just the methods we need to test
class TestModel:
    def __init__(self, output_path="./test_output"):
        self.output_path = Path(output_path)

    def _apply_nms_to_regions(self, binary_mask, min_region_size=50):
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
        # Create a copy of the mask to draw lines on
        line_mask = binary_mask.copy()

        # Detect edges using Canny (if not already edges)
        edges = cv2.Canny(binary_mask.astype(np.uint8) * 255, 50, 150)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        # If lines were found, draw them on the mask
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw thicker lines to create ROI regions around detected lines
                cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness=5)

        # Combine the original mask with the line mask
        combined_mask = np.maximum(binary_mask, line_mask)

        return combined_mask

    def generate_roi_mask_from_input(self, input_image, threshold_ratio=0.03):
        """
        Generate ROI mask directly from the input image.
        
        Args:
            input_image (np.ndarray): The input MRI image
            threshold_ratio (float): Value between 0 and 1 for thresholding
        
        Returns:
            np.ndarray: Binary mask where 1 indicates ROI regions
        """
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


def test_input_roi_generation():
    # Create a test instance
    model = TestModel(output_path="./test_input_roi")

    # Create a synthetic input image for testing
    h, w = 320, 320
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x * x + y * y)

    # Create a circular feature with a sharp edge
    input_image = np.zeros((h, w))
    input_image[d < 0.5] = 1.0

    # Add another feature (like a lesion or structure)
    input_image[100:150, 100:150] = 0.5

    # Add some random noise to make it more realistic
    noise = np.random.normal(0, 0.05, input_image.shape)
    input_image = input_image + noise
    input_image = np.clip(input_image, 0, 1)

    # Create output directory
    output_dir = Path("./test_input_roi")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Test ROI mask generation with LOWER threshold values
    thresholds = [0.01, 0.05, 0.1, 0.2]
    roi_masks = []

    # Calculate gradient for debugging
    normalized_image = input_image / input_image.max()
    sobel_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Debug: Print the min, max, mean of gradient magnitude
    print(f"Gradient magnitude stats: min={gradient_magnitude.min():.5f}, max={gradient_magnitude.max():.5f}, mean={gradient_magnitude.mean():.5f}")

    # Generate ROI masks
    for threshold in thresholds:
        print(f"Processing threshold {threshold}...")

        # First just apply the threshold without any further processing
        binary_mask_raw = (gradient_magnitude > threshold * gradient_magnitude.max()).astype(np.uint8)

        # Generate the full ROI mask
        roi_mask = model.generate_roi_mask_from_input(input_image, threshold_ratio=threshold)
        roi_masks.append(roi_mask)

        # Save the raw thresholded mask for comparison
        cv2.imwrite(str(output_dir / f"raw_mask_threshold_{threshold}.png"), (binary_mask_raw * 255).astype(np.uint8))

    # Save input image
    cv2.imwrite(str(output_dir / "input_image.png"), (input_image * 255).astype(np.uint8))

    for i, mask in enumerate(roi_masks):
        cv2.imwrite(str(output_dir / f"roi_mask_threshold_{thresholds[i]}.png"), (mask * 255).astype(np.uint8))

    # Create gradient magnitude visualization
    cv2.imwrite(
        str(output_dir / "gradient_magnitude.png"),
        (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    )

    # Plot for visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')

    plt.subplot(2, 3, 2)
    plt.imshow(gradient_magnitude, cmap='jet')
    plt.title('Gradient Magnitude')

    for i, mask in enumerate(roi_masks):
        plt.subplot(2, 3, i + 3)
        plt.imshow(mask, cmap='gray')
        plt.title(f'ROI Mask (threshold={thresholds[i]})')

    plt.tight_layout()
    plt.savefig(str(output_dir / "input_roi_visualization.png"))
    plt.close()

    print(f"Test completed! Check the output in {output_dir.absolute()}")


if __name__ == "__main__":
    test_input_roi_generation()
