#!/usr/bin/env python3
"""
Simple script to warp a target image to align with a reference image.
Set the image paths in the variables below and run the script.
"""

import sys
import os
from termcolor import colored
import torch

# Add the src directory to the path so we can import cruxes modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from imm import get_matcher
from cruxes.utils.warp_video import warp_image_to_reference


def main():
    # ========================================
    # SET YOUR IMAGE PATHS HERE
    # ========================================

    # Path to your reference image (the target alignment)
    reference_image_path = "examples/videos/warp-dynamic-ref.jpg"

    # Path to your target image (the image to be warped)
    target_image_path = "examples/videos/current_frame.jpg"

    # Path where you want to save the warped result
    output_image_path = "output/warped_result.jpg"

    # Optional: Add text overlay to the warped image
    overlay_text = True
    custom_text = "Warped Image"  # Set to None to use filename instead

    # ========================================
    # PROCESSING
    # ========================================

    print(colored("=== Image Warping Script ===", "cyan"))
    print(colored(f"Reference Image: {reference_image_path}", "blue"))
    print(colored(f"Target Image: {target_image_path}", "blue"))
    print(colored(f"Output Image: {output_image_path}", "blue"))

    # Check if input files exist
    if not os.path.exists(reference_image_path):
        print(
            colored(f"Error: Reference image not found: {reference_image_path}", "red")
        )
        return

    if not os.path.exists(target_image_path):
        print(colored(f"Error: Target image not found: {target_image_path}", "red"))
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(colored(f"Created output directory: {output_dir}", "green"))

    # Setup device (prefer MPS for Apple Silicon, fallback to CPU)
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            print(colored("Using MPS (Apple Silicon) for acceleration", "green"))
    except Exception as e:
        print(colored("MPS not available, using CPU", "yellow"))

    # Initialize the matcher
    print(colored("Initializing feature matcher...", "yellow"))
    matcher = get_matcher(
        "superpoint-lg",  # You can change this to other models like "superglue", "d2-net", "r2d2"
        device=device,
    )

    # Perform the image warping
    print(colored("Warping image...", "yellow"))
    success = warp_image_to_reference(
        reference_image_path=reference_image_path,
        target_image_path=target_image_path,
        output_image_path=output_image_path,
        matcher=matcher,
        overlay_text=overlay_text,
        text_to_overlay=custom_text,
    )

    if success:
        print(colored("✓ Image warping completed successfully!", "green"))
        print(colored(f"✓ Warped image saved to: {output_image_path}", "green"))
    else:
        print(colored("✗ Image warping failed!", "red"))


if __name__ == "__main__":
    main()
