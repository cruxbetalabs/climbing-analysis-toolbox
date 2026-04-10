import sys
import os

# Add the local src directory to the path BEFORE any other imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from imm import get_matcher
from cruxes.utils.warp_video import warp_image_to_reference


def main():

    matcher = get_matcher(
        # Available models:
        # https://github.com/alexstoken/image-matching-models?tab=readme-ov-file#available-models
        "superpoint-lightglue",  # "superglue",  # "d2-net", "r2d2"
        device="mps",
    )

    reference_image = "/Users/tommyjtl/Documents/Crux Beta Labs LLC/projects/spray/public/temporary_static_assets/example-problems/benchmark-bkl/iusjdnf.jpg"
    target_image_path = "/Users/tommyjtl/Documents/Crux Beta Labs LLC/projects/spray/public/temporary_static_assets/example-problems/benchmark-bkl/3.png"
    output_image_path = "/Users/tommyjtl/Documents/Crux Beta Labs LLC/projects/spray/public/temporary_static_assets/example-problems/benchmark-bkl/out1.png"

    warp_image_to_reference(
        reference_image,
        target_image_path,
        output_image_path,
        matcher,
        use_gradient_blending=False,
    )


if __name__ == "__main__":
    main()
