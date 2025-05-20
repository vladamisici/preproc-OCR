#!/usr/bin/env python3
"""
DocumentDenoise CLI Tool

This tool provides document denoising functionality using two methods:
1. Auto-encoder (AE) method
2. CycleGAN (CG) method

The tool preserves 100% of the logic from the original Jupyter notebooks.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document Denoising CLI Tool')
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True, help='Path to input image or directory of images')
    parser.add_argument('--output', '-o', required=True, help='Path to output directory')
    
    # Optional arguments
    parser.add_argument('--method', '-m', choices=['ae', 'cg'], default='ae',
                        help='Denoising method: ae (Auto-encoder) or cg (CycleGAN). Default: ae')
    parser.add_argument('--checkpoint', '-c', help='Path to checkpoint directory. If not provided, default checkpoints will be used')
    parser.add_argument('--display', '-d', action='store_true', help='Display before/after images')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()

def process_image_ae(img_path, img_width=3024, img_height=4032):
    """Process image for Auto-encoder method."""
    img = cv2.imread(img_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (img_height, img_width, 1))
    return img

def process_image_cg(img_path, img_width=3072, img_height=4096):
    """Process image for CycleGAN method."""
    img = cv2.imread(img_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (img_height, img_width, 1))
    return img

def create_ae_model(img_height=4032, img_width=3024):
    """Create Auto-encoder model."""
    input_layer = Input(shape=(img_height, img_width, 1))
    
    # encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Dropout(0.5)(x)
    
    # decoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(x)
    
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    
    return model

def denoise_with_ae(input_path, output_path, checkpoint_path=None, display=False, verbose=False):
    """Denoise document using Auto-encoder method."""
    if verbose:
        print(f"Using Auto-encoder method for denoising")
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'checkpoints/autoencoders/checkpoint_kaggle_80eps')
    
    # Check if input is a directory or a file
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    else:
        input_files = [input_path]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create and load model
    model = create_ae_model()
    
    try:
        if verbose:
            print(f"Loading model weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please make sure the checkpoint file exists and is valid.")
        sys.exit(1)
    
    # Process each input file
    for i, img_path in enumerate(input_files):
        if verbose:
            print(f"Processing image {i+1}/{len(input_files)}: {img_path}")
        
        # Process image
        img = process_image_ae(img_path)
        
        # Predict (denoise)
        denoised_img = model.predict(np.expand_dims(img, axis=0), verbose=0, batch_size=1)[0]
        
        # Save output
        base_name = os.path.basename(img_path)
        output_file = os.path.join(output_path, f"denoised_{base_name}")
        
        # Convert to 8-bit image for saving
        denoised_img_8bit = (denoised_img[:,:,0] * 255).astype(np.uint8)
        cv2.imwrite(output_file, denoised_img_8bit)
        
        if verbose:
            print(f"Saved denoised image to {output_file}")
        
        # Display if requested
        if display:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(img[:,:,0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title('Denoised')
            plt.imshow(denoised_img[:,:,0], cmap='gray')
            plt.show()
    
    return len(input_files)

def denoise_with_cg(input_path, output_path, checkpoint_path=None, display=False, verbose=False):
    """Denoise document using CycleGAN method."""
    if verbose:
        print(f"Using CycleGAN method for denoising")
    
    # Import tensorflow_examples for pix2pix model
    try:
        from tensorflow_examples.models.pix2pix import pix2pix
    except ImportError:
        print("Error: tensorflow_examples package not found.")
        print("Please install it using: pip install git+https://github.com/tensorflow/examples.git")
        sys.exit(1)
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'checkpoints/cyclegan/checkpoint')
    
    # Check if input is a directory or a file
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    else:
        input_files = [input_path]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create generator model
    generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
    
    try:
        if verbose:
            print(f"Loading model checkpoint from {checkpoint_path}")
        # Load the checkpoint
        checkpoint = tf.train.Checkpoint(generator_g=generator_g)
        checkpoint.restore(checkpoint_path).expect_partial()
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        print("Please make sure the checkpoint file exists and is valid.")
        sys.exit(1)
    
    # Process each input file
    for i, img_path in enumerate(input_files):
        if verbose:
            print(f"Processing image {i+1}/{len(input_files)}: {img_path}")
        
        # Process image
        img = process_image_cg(img_path)
        
        # Convert to 3-channel for CycleGAN
        img_3channel = np.concatenate([img, img, img], axis=2)
        
        # Normalize to [-1, 1] for CycleGAN
        img_3channel = (img_3channel * 2) - 1
        
        # Predict (denoise)
        denoised_img = generator_g(np.expand_dims(img_3channel, axis=0), training=False)[0]
        
        # Convert back to [0, 1] range
        denoised_img = (denoised_img + 1) / 2.0
        
        # Save output
        base_name = os.path.basename(img_path)
        output_file = os.path.join(output_path, f"denoised_{base_name}")
        
        # Convert to 8-bit image for saving
        denoised_img_8bit = (np.mean(denoised_img, axis=2) * 255).astype(np.uint8)
        cv2.imwrite(output_file, denoised_img_8bit)
        
        if verbose:
            print(f"Saved denoised image to {output_file}")
        
        # Display if requested
        if display:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(img[:,:,0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title('Denoised')
            plt.imshow(np.mean(denoised_img, axis=2), cmap='gray')
            plt.show()
    
    return len(input_files)

def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)
    
    # Choose denoising method
    if args.method == 'ae':
        num_processed = denoise_with_ae(
            args.input, 
            args.output, 
            args.checkpoint, 
            args.display, 
            args.verbose
        )
    else:  # args.method == 'cg'
        num_processed = denoise_with_cg(
            args.input, 
            args.output, 
            args.checkpoint, 
            args.display, 
            args.verbose
        )
    
    print(f"\nDenoising completed successfully!")
    print(f"Method: {'Auto-encoder' if args.method == 'ae' else 'CycleGAN'}")
    print(f"Images processed: {num_processed}")
    print(f"Output saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
