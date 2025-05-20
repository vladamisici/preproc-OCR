#!/usr/bin/env python3
"""
DocumentDenoise CLI Tool - Test Script

This script tests the DocumentDenoise CLI tool with sample images.
"""

import os
import sys
import argparse
import shutil
from denoise import denoise_with_ae, denoise_with_cg

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test DocumentDenoise CLI tool')
    
    parser.add_argument('--test-dir', '-t', default='test_images',
                        help='Directory to store test images (default: test_images)')
    parser.add_argument('--method', '-m', choices=['ae', 'cg', 'both'], default='both',
                        help='Method to test: ae, cg, or both (default: both)')
    
    return parser.parse_args()

def setup_test_environment(test_dir):
    """Set up test environment with sample images."""
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Create output directory
    output_dir = os.path.join(test_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if there are any images in the test directory
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not image_files:
        print("No test images found. Please add some test images to the test directory.")
        print(f"Test directory: {os.path.abspath(test_dir)}")
        return False
    
    return True

def run_tests(test_dir, method):
    """Run tests with the specified method."""
    output_dir = os.path.join(test_dir, 'output')
    
    # Get test images
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    # Test Auto-encoder method
    if method in ['ae', 'both']:
        print("\n=== Testing Auto-encoder Method ===")
        ae_output_dir = os.path.join(output_dir, 'ae')
        os.makedirs(ae_output_dir, exist_ok=True)
        
        for img_file in image_files:
            img_path = os.path.join(test_dir, img_file)
            print(f"Processing {img_file} with Auto-encoder method...")
            try:
                denoise_with_ae(img_path, ae_output_dir, display=False, verbose=True)
                print(f"Successfully processed {img_file} with Auto-encoder method")
            except Exception as e:
                print(f"Error processing {img_file} with Auto-encoder method: {e}")
    
    # Test CycleGAN method
    if method in ['cg', 'both']:
        print("\n=== Testing CycleGAN Method ===")
        cg_output_dir = os.path.join(output_dir, 'cg')
        os.makedirs(cg_output_dir, exist_ok=True)
        
        for img_file in image_files:
            img_path = os.path.join(test_dir, img_file)
            print(f"Processing {img_file} with CycleGAN method...")
            try:
                denoise_with_cg(img_path, cg_output_dir, display=False, verbose=True)
                print(f"Successfully processed {img_file} with CycleGAN method")
            except Exception as e:
                print(f"Error processing {img_file} with CycleGAN method: {e}")
    
    print("\n=== Test Summary ===")
    print(f"Test directory: {os.path.abspath(test_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Methods tested: {method}")
    print(f"Images processed: {len(image_files)}")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("=== DocumentDenoise CLI Test ===")
    
    # Setup test environment
    if not setup_test_environment(args.test_dir):
        sys.exit(1)
    
    # Run tests
    run_tests(args.test_dir, args.method)
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
