import numpy as np
import argparse
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.stats import mode
import os

def detect_lines(image, dir, show_plot=False):
    """
    Detect lines in an image using Hough transform.
    
    Args:
        image: Input grayscale image
        dir: Direction of lines to detect ('hor' for horizontal, 'ver' for vertical)
        show_plot: Whether to display the plot of detected lines
    """
    sigma = 1
    edges = canny(image, sigma=sigma)
    
    # Set angle range based on direction
    if dir == "hor":
        tested_angles = np.deg2rad(np.arange(75, 105))
    else:
        tested_angles = np.deg2rad(np.arange(165, 195))
    
    h, theta, d = hough_line(edges, theta=tested_angles)

    if show_plot:
        # Generate visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 16))
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(edges, cmap="gray")
        origin = np.array((0, image.shape[1]))

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[1].plot(origin, (y0, y1), '-r')
            
        ax[1].set_xlim(origin)
        ax[1].set_ylim((edges.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Detected lines')
        plt.show()

def skew_angle_hough_transform(image, sigma=1):
    """
    Determine the skew angle of an image using Hough transform.
    
    Args:
        image: Input grayscale image
        sigma: Sigma parameter for Canny edge detection
        
    Returns:
        Detected skew angle in degrees
    """
    # Convert to edges
    edges = canny(image, sigma=sigma)
    
    # Classic straight-line Hough transform
    tested_angles = np.deg2rad(np.arange(75, 105))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # Find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    if len(angles) == 0:
        print("No lines detected. Using 0 as skew angle.")
        return 0
    
    # Round the angles to 2 decimal places and find the most common angle
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # Convert the angle to degree for rotation
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    
    return skew_angle

def deskew_image(image_path, output_path=None, show_result=False, sigma=1):
    """
    Deskew an image and save the result.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, uses input path with _deskewed suffix)
        show_result: Whether to display the before/after comparison
        sigma: Sigma parameter for Canny edge detection
        
    Returns:
        Path to the saved deskewed image
    """
    # Load image
    image = imread(image_path)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image / 255.0 if image.max() > 1 else image
    
    # Calculate skew angle
    skew_angle = skew_angle_hough_transform(gray_image, sigma=sigma)
    print(f"Detected skew angle: {skew_angle:.2f} degrees")
    
    # Rotate image to correct skew
    if abs(skew_angle) > 0.1:  # Only rotate if angle is significant
        # For color images, rotate each channel
        if len(image.shape) == 3:
            rotated = np.zeros_like(image)
            for i in range(image.shape[2]):
                rotated[:,:,i] = rotate(image[:,:,i], skew_angle, resize=True, 
                                        mode='constant', cval=255, preserve_range=True)
            rotated = rotated.astype(image.dtype)
        else:
            rotated = rotate(image, skew_angle, resize=True, mode='constant', 
                             cval=255, preserve_range=True).astype(image.dtype)
    else:
        print("Skew angle is negligible. No rotation performed.")
        rotated = image
    
    # Generate output path if not provided
    if output_path is None:
        file_name, file_ext = os.path.splitext(image_path)
        output_path = f"{file_name}_deskewed{file_ext}"
    
    # Save deskewed image
    imsave(output_path, rotated)
    print(f"Deskewed image saved to: {output_path}")
    
    # Show result if requested
    if show_result:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
        ax[0].imshow(image, cmap="gray" if len(image.shape) == 2 else None)
        ax[1].imshow(rotated, cmap="gray" if len(rotated.shape) == 2 else None)
        ax[0].set_title('Input image')
        ax[1].set_title('Deskewed image')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.tight_layout()
        plt.show()
    
    return output_path

def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description='Deskew document images using Hough transform.')
    parser.add_argument('--input', '-i', required=True, help='Path to input image or directory')
    parser.add_argument('--output', '-o', help='Path to output image or directory (optional)')
    parser.add_argument('--show', '-s', action='store_true', help='Show result comparison')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Canny edge detection (default: 1.0)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
    
    args = parser.parse_args()
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        if args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        
        # Process all images in directory
        for root, _, files in os.walk(args.input) if args.recursive else [(args.input, None, os.listdir(args.input))]:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    input_path = os.path.join(root, file)
                    
                    if args.output:
                        rel_path = os.path.relpath(input_path, args.input)
                        output_path = os.path.join(args.output, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = None
                    
                    try:
                        print(f"Processing: {input_path}")
                        deskew_image(input_path, output_path, args.show, args.sigma)
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")
    else:
        # Process single image
        deskew_image(args.input, args.output, args.show, args.sigma)

if __name__ == "__main__":
    main()