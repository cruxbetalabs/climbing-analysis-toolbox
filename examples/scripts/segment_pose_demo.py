import argparse
from termcolor import colored

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cruxes import Cruxes


def main():
    parser = argparse.ArgumentParser(
        description="Segment person from video and replace background with solid color."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default="0,255,0",
        help="Background color in BGR format (comma-separated, e.g., '0,255,0' for green).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation confidence threshold (0.0 to 1.0). Higher values = stricter segmentation.",
    )
    args = parser.parse_args()
    
    if not args.video_path or args.video_path == "":
        print(
            colored(
                "Warning: No video path supplied. Please provide --video_path.",
                "red",
            )
        )
        return
    
    target_video_path = args.video_path
    
    # Parse background color
    try:
        bg_color = tuple(map(int, args.background_color.split(',')))
        if len(bg_color) != 3:
            raise ValueError("Background color must have 3 values (B,G,R)")
        if not all(0 <= c <= 255 for c in bg_color):
            raise ValueError("Color values must be between 0 and 255")
    except Exception as e:
        print(
            colored(
                f"Error parsing background color: {e}. Using default green (0,255,0).",
                "red",
            )
        )
        bg_color = (0, 255, 0)

    # Print colored messages for debugging
    print(colored("Target video path:", "blue"), target_video_path)
    print(colored("Background color (BGR):", "blue"), bg_color)
    print(colored("Segmentation threshold:", "blue"), args.threshold)

    cruxes = Cruxes()
    cruxes.segment_pose(
        target_video_path,
        background_color=bg_color,
        segmentation_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
