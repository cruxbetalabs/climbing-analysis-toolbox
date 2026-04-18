import argparse
import sys
import os
from termcolor import colored

# Add the local src directory to the path BEFORE any other imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from cruxes import Cruxes


def main():
    parser = argparse.ArgumentParser(
        description="Warp an image based on a reference image."
    )
    parser.add_argument(
        "--src_img_path",
        type=str,
        default=None,
        help="Path to the source image file.",
    )
    parser.add_argument(
        "--ref_img",
        type=str,
        default=None,
        help="Path to the reference image file.",
    )
    parser.add_argument(
        "--output_img_path",
        type=str,
        default=None,
        help="Optional output image path. Defaults to warped_<input_name> next to the source image.",
    )
    parser.add_argument(
        "--overlay_text",
        action="store_true",
        default=False,
        help="Whether to overlay text on the warped image.",
    )
    parser.add_argument(
        "--text_to_overlay",
        type=str,
        default=None,
        help="Optional text to overlay on the warped image.",
    )
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="edge_feather",
        choices=["none", "feathered", "edge_feather", "smart", "multiband", "poisson"],
        help="Blending mode for compositing the warped image.",
    )
    parser.add_argument(
        "--feather_amount",
        type=int,
        default=15,
        help="Feathering amount in pixels for supported blend modes.",
    )
    parser.add_argument(
        "--use_gradient_blending",
        action="store_true",
        default=False,
        help="Use the legacy Poisson blending path. Prefer --blend_mode poisson.",
    )
    parser.add_argument(
        "--matcher_model_name",
        type=str,
        default="superpoint-lightglue",
        help="Matcher model name from the upstream vismatch/imm model registry.",
    )
    parser.add_argument(
        "--matcher_device",
        type=str,
        default="auto",
        help="Matcher device override. Use auto, cpu, mps, or cuda.",
    )
    args = parser.parse_args()

    if not args.src_img_path or not args.ref_img:
        print(
            colored(
                "Warning: No source image path or reference image supplied. Please provide --src_img_path and --ref_img.",
                "red",
            )
        )
        return

    print(colored(f"Reference Image: {args.ref_img}", "blue"))
    print(colored(f"Source Image: {args.src_img_path}", "blue"))
    if args.output_img_path:
        print(colored(f"Output Image: {args.output_img_path}", "blue"))
    print(colored(f"Blend Mode: {args.blend_mode}", "blue"))
    print(colored(f"Matcher Model: {args.matcher_model_name}", "blue"))
    print(colored(f"Matcher Device: {args.matcher_device}", "blue"))

    cruxes = Cruxes(
        matcher_model_name=args.matcher_model_name,
        # matcher_device="cpu",
    )
    cruxes.warp_image(
        args.ref_img,
        args.src_img_path,
        output_image_path=args.output_img_path,
        overlay_text=args.overlay_text,
        text_to_overlay=args.text_to_overlay,
        use_gradient_blending=args.use_gradient_blending,
        blend_mode=args.blend_mode,
        feather_amount=args.feather_amount,
    )


if __name__ == "__main__":
    main()
