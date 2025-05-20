#!/usr/bin/env python3
"""
DocumentDenoise CLI Tool - Directory Structure Setup

This script creates the necessary directory structure for the DocumentDenoise CLI tool.
It creates the checkpoints directory and subdirectories for both denoising methods.
"""

import os
import sys
import shutil
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup DocumentDenoise CLI directory structure')
    
    parser.add_argument('--source', '-s', help='Path to source project with checkpoints (optional)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create checkpoints directory and subdirectories
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')
    ae_dir = os.path.join(checkpoints_dir, 'autoencoders')
    cg_dir = os.path.join(checkpoints_dir, 'cyclegan')
    
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(cg_dir, exist_ok=True)
    
    print(f"Created directory structure:")
    print(f"- {checkpoints_dir}")
    print(f"  - {ae_dir}")
    print(f"  - {cg_dir}")
    
    # If source directory is provided, try to copy checkpoints
    if args.source:
        if not os.path.exists(args.source):
            print(f"Error: Source directory '{args.source}' does not exist.")
            sys.exit(1)
        
        # Try to find and copy Auto-encoder checkpoints
        ae_source = os.path.join(args.source, 'checkpoints/autoencoders/checkpoint_kaggle_80eps')
        if os.path.exists(ae_source):
            shutil.copy(ae_source, os.path.join(ae_dir, 'checkpoint_kaggle_80eps'))
            print(f"Copied Auto-encoder checkpoint from {ae_source}")
        else:
            print(f"Warning: Auto-encoder checkpoint not found at {ae_source}")
        
        # Try to find and copy CycleGAN checkpoints
        cg_source = os.path.join(args.source, 'checkpoints/cyclegan/checkpoint')
        if os.path.exists(cg_source):
            shutil.copy(cg_source, os.path.join(cg_dir, 'checkpoint'))
            print(f"Copied CycleGAN checkpoint from {cg_source}")
        else:
            print(f"Warning: CycleGAN checkpoint not found at {cg_source}")
    
    print("\nSetup completed. You may need to manually copy checkpoint files if not automatically found.")
    print("The DocumentDenoise CLI tool is now ready to use.")

if __name__ == "__main__":
    main()
