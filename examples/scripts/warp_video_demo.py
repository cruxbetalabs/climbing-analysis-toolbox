import argparse
from termcolor import colored

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cruxes import Cruxes


def main():
    parser = argparse.ArgumentParser(
        description="Warp video based on a reference image."
    )
    parser.add_argument(
        "--src_video_path",
        type=str,
        default=None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--ref_img",
        type=str,
        default=None,
        help="Path to the reference image file.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="dynamic",
        help="Type of homography use: fixed or dynamic",
    )
    parser.add_argument(
        "--overlay_text",
        type=bool,
        default=False,
        help="Whether to overlay text on the video frames",
    )
    args = parser.parse_args()

    # Check if required arguments are provided
    # If not, print a warning and return
    if not args.src_video_path or not args.ref_img:
        print(
            colored(
                "Warning: No video path or reference image supplied. Please provide --src_video_path and --ref_img",
                "red",
            )
        )
        return

    reference_image = args.ref_img
    target_video = args.src_video_path
    warp_type = args.type
    overlay_text = args.overlay_text

    # Print colored messages for debugging
    print(colored(f"Reference Image: {reference_image}", "blue"))
    print(colored(f"Target Video: {target_video}", "blue"))
    print(colored(f"Warp Type: {warp_type}", "blue"))
    print(colored(f"Overlay Text: {overlay_text}", "blue"))

    cruxes = Cruxes(matcher_device="mps")
    # Blending modes to avoid shadowing:
    # - 'edge_feather': Best for avoiding shadows, only blends at edges (RECOMMENDED)
    # - 'smart': Clean morphological blending, very minimal shadowing
    # - 'multiband': High quality pyramid blending, slower but professional results
    # Lower feather_amount (5-10) = less shadowing, sharper edges
    # Higher feather_amount (15-30) = more blending, softer but may show shadows
    cruxes.warp_video(
        reference_image,
        target_video,
        warp_type,
        overlay_text,
        blend_mode="none",
        # Options: 'none', 'feathered', 'edge_feather', 'smart', 'multiband', 'poisson'
    )


if __name__ == "__main__":
    main()
