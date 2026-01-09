import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
import tempfile


def tensor_to_np(img):
    """
    Convert a torch tensor image (CHW) to a normalized numpy array (HWC, uint8).
    """
    if torch.is_tensor(img):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        return img_np
    return img  # fallback if not tensor


def add_text_to_image(
    image,
    text,
    position=(10, 60),
    font_scale=2.0,
    color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=3,
):
    """
    Overlay text on an image at a given position.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bg_pos = (position[0] + 2, position[1] + 2)
    cv2.putText(image, text, bg_pos, font, font_scale, bg_color, thickness, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust the brightness and contrast of an image.
    alpha: Contrast control (1.0 means no change).
    beta: Brightness control (0 means no change).
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def compute_image_contrast(image):
    """
    Compute a simple contrast measure of an image using std of grayscale intensities.
    """
    return np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def save_frame_as_jpg(frame, path):
    """
    Save a video frame as a JPEG file.
    """
    cv2.imwrite(path, frame)


def load_and_prepare_images(reference_path, frame, matcher, temp_frame_path):
    """
    Load reference and target images using matcher loader.
    """
    img_reference = matcher.load_image(reference_path)
    save_frame_as_jpg(frame, temp_frame_path)
    img_target = matcher.load_image(temp_frame_path)
    return img_reference, img_target


def invert_homography(H):
    """
    Invert a homography matrix if not None.
    """
    if H is not None:
        return np.linalg.inv(H)
    return None


def warp_image(img, H, shape):
    """
    Warp an image using a homography matrix to a given shape.
    """
    return cv2.warpPerspective(img, H, shape)


def create_white_canvas_like(img):
    """
    Create a white canvas with the same shape as the input image.
    """
    return np.ones_like(img) * 255


def apply_gradient_blending(background, foreground, mask):
    """
    Apply Poisson blending (mixed gradient) for seamless image composition.

    Args:
        background: Background image (reference image)
        foreground: Foreground image (warped frame)
        mask: Binary mask indicating the foreground region

    Returns:
        Blended image with seamless transitions
    """
    # Find the center of the non-zero region in the mask
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    coords = cv2.findNonZero(mask_gray)

    if coords is None or len(coords) == 0:
        # No valid region to blend, return background
        return background

    # Calculate center point for seamless cloning
    x, y, w, h = cv2.boundingRect(coords)
    center = (x + w // 2, y + h // 2)

    # Ensure images are in the correct format (8-bit, 3-channel)
    bg = background.astype(np.uint8)
    fg = foreground.astype(np.uint8)
    mask_8bit = mask_gray.astype(np.uint8)

    try:
        # Use MIXED_CLONE for gradient domain blending
        # This preserves gradients from both source and destination
        blended = cv2.seamlessClone(fg, bg, mask_8bit, center, cv2.MIXED_CLONE)
        return blended
    except cv2.error as e:
        print(
            f"Warning: Gradient blending failed ({e}), falling back to standard blending"
        )
        return combine_images_with_masks(background, foreground, mask)


def apply_feathered_blending(background, foreground, mask, feather_amount=15):
    """
    Apply feathered alpha blending for smooth transitions while preserving foreground opacity.
    This method is more robust than seamlessClone and maintains foreground color/opacity better.

    Args:
        background: Background image (reference image)
        foreground: Foreground image (warped frame)
        mask: Binary mask indicating the foreground region
        feather_amount: Number of pixels to feather at the boundary (default: 15)

    Returns:
        Blended image with feathered edges
    """
    # Convert mask to grayscale if needed
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask

    # Create a binary mask (threshold to ensure clean binary)
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Apply Gaussian blur to create soft edges (feathering)
    # The kernel size should be odd and related to feather_amount
    kernel_size = max(3, feather_amount * 2 + 1)  # Ensure odd number
    feathered_mask = cv2.GaussianBlur(
        binary_mask.astype(np.float32), (kernel_size, kernel_size), 0
    )

    # Normalize to 0-1 range for alpha blending
    alpha = feathered_mask / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # Add channel dimension

    # Ensure images are in the correct format
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)

    # Alpha blend: result = foreground * alpha + background * (1 - alpha)
    blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

    return blended


def apply_edge_feathering(background, foreground, mask, feather_amount=15):
    """
    Apply edge-only feathering using distance transform.
    This only softens the boundary region without affecting the interior,
    preventing shadow-like artifacts.

    Args:
        background: Background image (reference image)
        foreground: Foreground image (warped frame)
        mask: Binary mask indicating the foreground region
        feather_amount: Number of pixels to feather at the boundary (default: 15)

    Returns:
        Blended image with only edge feathering
    """
    # Convert mask to grayscale if needed
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask

    # Create a binary mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Use distance transform to create a gradient only at the edges
    # Distance from the edge of the mask
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Create alpha mask that is:
    # - 1.0 for pixels more than feather_amount away from edge
    # - 0.0 for pixels outside the mask
    # - gradient from 0 to 1 within feather_amount pixels of the edge
    alpha = np.clip(dist_transform / feather_amount, 0, 1).astype(np.float32)

    # Apply slight smoothing to the alpha for smoother transitions
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.expand_dims(alpha, axis=2)  # Add channel dimension

    # Ensure images are in the correct format
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)

    # Alpha blend only at the edges
    blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

    return blended


def apply_smart_blending(background, foreground, mask, feather_amount=10):
    """
    Smart blending that preserves foreground and only blends at actual boundaries.
    Uses morphological operations to identify true edges and applies minimal blending.

    Args:
        background: Background image (reference image)
        foreground: Foreground image (warped frame)
        mask: Binary mask indicating the foreground region
        feather_amount: Number of pixels to feather at the boundary (default: 10)

    Returns:
        Blended image with smart edge handling
    """
    # Convert mask to grayscale if needed
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask

    # Create a binary mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Erode the mask to create an inner region (definitely foreground)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (feather_amount, feather_amount)
    )
    inner_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # The difference between original and eroded is the boundary region
    boundary_region = cv2.subtract(binary_mask, inner_mask)

    # Create smooth transition in the boundary region only
    boundary_dist = cv2.distanceTransform(boundary_region, cv2.DIST_L2, 5)
    boundary_alpha = np.clip(boundary_dist / (feather_amount / 2), 0, 1).astype(
        np.float32
    )

    # Combine: inner region is fully foreground, boundary has gradient, outside is background
    alpha = np.zeros_like(binary_mask, dtype=np.float32)
    alpha[inner_mask > 0] = 1.0
    alpha[boundary_region > 0] = boundary_alpha[boundary_region > 0]

    # Smooth the alpha slightly
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.expand_dims(alpha, axis=2)

    # Blend
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

    return blended


def apply_multiband_blending(background, foreground, mask, levels=4):
    """
    Apply multi-band (Laplacian pyramid) blending for high-quality seamless compositing.
    This preserves details better than simple alpha blending while still creating smooth transitions.

    Args:
        background: Background image (reference image)
        foreground: Foreground image (warped frame)
        mask: Binary mask indicating the foreground region
        levels: Number of pyramid levels (default: 4)

    Returns:
        Blended image using multi-band blending
    """
    # Convert mask to grayscale if needed
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Ensure images are the same size
    if background.shape != foreground.shape:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # Convert to float
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    mask_float = binary_mask.astype(np.float32) / 255.0

    # Build Gaussian pyramids for the mask
    gaussian_mask = [mask_float]
    for i in range(levels):
        mask_float = cv2.pyrDown(mask_float)
        gaussian_mask.append(mask_float)

    # Build Laplacian pyramids for both images
    def build_laplacian_pyramid(img, levels):
        gaussian_pyr = [img]
        for i in range(levels):
            img = cv2.pyrDown(img)
            gaussian_pyr.append(img)

        laplacian_pyr = []
        for i in range(levels):
            size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyr[i], upsampled)
            laplacian_pyr.append(laplacian)
        laplacian_pyr.append(gaussian_pyr[-1])  # Add the smallest level
        return laplacian_pyr

    laplacian_bg = build_laplacian_pyramid(bg, levels)
    laplacian_fg = build_laplacian_pyramid(fg, levels)

    # Blend each level of the pyramids
    blended_pyramid = []
    for i in range(len(laplacian_bg)):
        # Expand mask to match current level dimensions
        mask_level = gaussian_mask[min(i, len(gaussian_mask) - 1)]
        if mask_level.ndim == 2:
            mask_level = np.expand_dims(mask_level, axis=2)

        # Resize mask if needed to match pyramid level
        if laplacian_bg[i].shape[:2] != mask_level.shape[:2]:
            mask_level = cv2.resize(
                mask_level, (laplacian_bg[i].shape[1], laplacian_bg[i].shape[0])
            )
            if mask_level.ndim == 2:
                mask_level = np.expand_dims(mask_level, axis=2)

        # Blend at this pyramid level
        blended_level = laplacian_fg[i] * mask_level + laplacian_bg[i] * (
            1 - mask_level
        )
        blended_pyramid.append(blended_level)

    # Reconstruct the image from the blended pyramid
    blended = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        blended = cv2.add(blended, blended_pyramid[i])

    # Clip and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def combine_images_with_masks(
    R_prime,
    warped_img,
    warped_mask,
    use_gradient_blending=False,
    blend_mode="edge_feather",
    feather_amount=15,
):
    """
    Combine reference and warped images using masks to handle overlap and black areas.

    Args:
        R_prime: Reference image
        warped_img: Warped target image
        warped_mask: Mask from warping operation
        use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode instead)
        blend_mode: Blending mode - 'none', 'feathered', 'edge_feather', 'smart', 'multiband', or 'poisson'
                    (default: 'edge_feather')
        feather_amount: Feathering amount in pixels (default: 15)

    Returns:
        Combined image
    """
    # Handle backward compatibility
    if use_gradient_blending and blend_mode == "edge_feather":
        blend_mode = "poisson"

    if blend_mode in ["feathered", "edge_feather", "smart", "multiband", "poisson"]:
        # Create a binary mask for non-black regions in the warped image
        mask_non_black = cv2.inRange(warped_img, (1, 1, 1), (255, 255, 255))
        # Combine with warped_mask to get final mask
        final_mask = cv2.bitwise_and(mask_non_black, warped_mask[:, :, 0])

        # Apply the selected blending mode
        if blend_mode == "feathered":
            return apply_feathered_blending(
                R_prime, warped_img, final_mask, feather_amount
            )
        elif blend_mode == "edge_feather":
            return apply_edge_feathering(
                R_prime, warped_img, final_mask, feather_amount
            )
        elif blend_mode == "smart":
            return apply_smart_blending(R_prime, warped_img, final_mask, feather_amount)
        elif blend_mode == "multiband":
            return apply_multiband_blending(R_prime, warped_img, final_mask)
        elif blend_mode == "poisson":
            return apply_gradient_blending(R_prime, warped_img, final_mask)

    # Original masking-based combination (blend_mode='none' or default)
    R_double_prime = cv2.bitwise_or(
        R_prime, warped_img, mask=cv2.bitwise_not(warped_mask[:, :, 0])
    )

    mask_black_R_double_prime = cv2.inRange(R_double_prime, (0, 0, 0), (0, 0, 0))
    mask_black_warped_img = cv2.inRange(warped_img, (0, 0, 0), (0, 0, 0))
    mask_non_black_R_double_prime = cv2.bitwise_not(mask_black_R_double_prime)
    mask_non_black_warped_img = cv2.bitwise_not(mask_black_warped_img)

    O = cv2.bitwise_or(
        cv2.bitwise_and(
            R_double_prime, R_double_prime, mask=mask_non_black_R_double_prime
        ),
        cv2.bitwise_and(warped_img, warped_img, mask=mask_black_R_double_prime),
    )
    O = cv2.bitwise_or(
        O,
        cv2.bitwise_and(warped_img, warped_img, mask=mask_non_black_warped_img),
    )
    O = cv2.bitwise_or(
        O,
        cv2.bitwise_and(R_double_prime, R_double_prime, mask=mask_black_warped_img),
    )

    return O


def normalize_and_convert_color(img, convert=True):
    """
    Normalize image to 8-bit and convert BGR to RGB.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if convert == True:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_warped_frame_using_diff_H(
    img_reference_path,
    frame,
    matcher,
    parent_dir,
    use_gradient_blending=False,
    blend_mode="none",
    feather_amount=15,
):
    """
    Warp a video frame to align with a reference image using feature matching and homography.

    Args:
        img_reference_path: Path to reference image
        frame: Video frame to warp
        matcher: Feature matcher object
        parent_dir: Directory for temporary files (deprecated, uses system temp directory)
        use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode)
        blend_mode: Blending mode - 'none', 'feathered', 'edge_feather', 'smart', 'multiband', or 'poisson'
        feather_amount: Feathering amount in pixels

    Returns:
        Warped frame aligned to reference image
    """
    # Create a unique temporary file for this frame to avoid conflicts in parallel execution
    temp_fd, temp_frame_path = tempfile.mkstemp(suffix=".jpg", prefix="current_frame_")
    os.close(temp_fd)  # Close the file descriptor, we just need the path
    # print(f"Using temporary frame path: {temp_frame_path}")

    try:
        img_reference, img_target = load_and_prepare_images(
            img_reference_path, frame, matcher, temp_frame_path
        )
        result = matcher(img_reference, img_target)
        H = result["H"]

        img0_np = tensor_to_np(img_reference)
        img1_np = tensor_to_np(img_target)

        if H is not None:
            H_inv = invert_homography(H)
            h, w = img0_np.shape[:2]
            warped_img1_np = warp_image(img1_np, H_inv, (w, h))
            white_canvas = create_white_canvas_like(img1_np)
            warped_mask = warp_image(white_canvas, H_inv, (w, h))
            R_prime = img0_np.copy()
            O = combine_images_with_masks(
                R_prime,
                warped_img1_np,
                warped_mask,
                use_gradient_blending,
                blend_mode,
                feather_amount,
            )
            O = normalize_and_convert_color(O)
            return O

        print("Homography matrix is None. Cannot warp the frame.")
        return np.zeros_like(frame)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)


def compute_homography_once(img_reference_path, target_image_path, matcher):
    """
    Compute homography between reference image and a single target image.
    """
    img_reference = matcher.load_image(img_reference_path)
    img_target = matcher.load_image(target_image_path)
    result = matcher(img_reference, img_target)
    H = result["H"]
    return H, img_reference, img_target


def warp_video_with_fixed_homography(
    img_reference_path,
    target_video_path,
    matcher,
    parent_dir,
    output_video_path,
    overlay_text=True,
    use_gradient_blending=False,
    blend_mode="edge_feather",
    feather_amount=15,
):
    """
    Warp video using a fixed homography computed from the first frame.

    Args:
        img_reference_path: Path to reference image
        target_video_path: Path to input video
        matcher: Feature matcher object
        parent_dir: Directory for temporary files
        output_video_path: Path for output video
        overlay_text: Whether to overlay filename on frames
        use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode)
        blend_mode: Blending mode - 'none', 'feathered', 'edge_feather', 'smart', 'multiband', or 'poisson'
        feather_amount: Feathering amount in pixels
    """
    file_name = os.path.splitext(os.path.basename(target_video_path))[0]
    ref_img = cv2.imread(img_reference_path)
    ref_height, ref_width = ref_img.shape[:2]
    print(f"Reference image dimensions: {ref_width} x {ref_height}")

    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Target video dimensions: {video_width} x {video_height}")

    # Use output_video_path instead of saving to parent_dir
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref_width, ref_height))

    # Extract the first frame from the video and save as temp image
    cap_first = cv2.VideoCapture(target_video_path)
    ret_first, first_frame = cap_first.read()
    if not ret_first:
        print("Error: Could not read the first frame from video.")
        return
    # if parent_dir is empty, use current directory
    if not parent_dir:
        parent_dir = os.getcwd()
    temp_first_frame_path = f"{parent_dir}/_temp_first_frame.jpg"
    cv2.imwrite(temp_first_frame_path, first_frame)
    cap_first.release()

    # Compute homography once using the first frame as target image
    H, img_reference, img_target = compute_homography_once(
        img_reference_path, temp_first_frame_path, matcher
    )
    if H is not None:
        H_inv = invert_homography(H)
    else:
        print("Homography matrix is None. Cannot warp the video.")
        return

    # Ensure img_reference is in BGR for consistency with OpenCV
    R_prime_bgr = tensor_to_np(img_reference)
    if R_prime_bgr.shape[2] == 3:
        R_prime_bgr = cv2.cvtColor(R_prime_bgr, cv2.COLOR_RGB2BGR)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Determine blending mode for display
        if use_gradient_blending and blend_mode == "edge_feather":
            display_mode = "poisson"
        else:
            display_mode = blend_mode

        with tqdm(
            total=total_frames,
            desc=f"Processing Video (fixed H, {display_mode} blending)",
            unit="frame",
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Finished reading the video.")
                    break
                img1_np = tensor_to_np(frame)

                # Add file name text to the top left corner of the target frame (after warped, not combined)
                if overlay_text:
                    img1_np_with_text = add_text_to_image(img1_np.copy(), file_name)
                else:
                    img1_np_with_text = img1_np.copy()

                warped_img1_np = warp_image(
                    img1_np_with_text, H_inv, (ref_width, ref_height)
                )
                white_canvas = create_white_canvas_like(img1_np)
                warped_mask = warp_image(white_canvas, H_inv, (ref_width, ref_height))
                # Use BGR reference for combining
                O = combine_images_with_masks(
                    R_prime_bgr,
                    warped_img1_np,
                    warped_mask,
                    use_gradient_blending,
                    blend_mode,
                    feather_amount,
                )
                O = normalize_and_convert_color(O, convert=False)  # keep BGR for OpenCV
                if O.shape[:2] != (ref_height, ref_width):
                    O = cv2.resize(O, (ref_width, ref_height))
                pbar.update(1)
                out.write(O)
            cv2.destroyAllWindows()
    cap.release()
    out.release()


def warp_image_to_reference(
    reference_image_path,
    target_image_path,
    output_image_path,
    matcher,
    overlay_text=False,
    text_to_overlay=None,
    use_gradient_blending=False,
    blend_mode="edge_feather",
    feather_amount=15,
):
    """
    Warp a target JPEG image to align with a reference JPEG image using feature matching and homography.

    Args:
        reference_image_path (str): Path to the reference image
        target_image_path (str): Path to the target image to be warped
        output_image_path (str): Path where the warped image will be saved
        matcher: Feature matcher object (e.g., LightGlue matcher)
        overlay_text (bool): Whether to overlay text on the warped image
        text_to_overlay (str): Text to overlay (if None, uses target filename)
        use_gradient_blending (bool): If True, use advanced blending (deprecated, use blend_mode)
        blend_mode (str): Blending mode - 'none', 'feathered', 'edge_feather', 'smart', 'multiband', or 'poisson'
        feather_amount (int): Feathering amount in pixels

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load reference image to get dimensions
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            print(f"Error: Could not load reference image from {reference_image_path}")
            return False

        ref_height, ref_width = ref_img.shape[:2]
        print(f"Reference image dimensions: {ref_width} x {ref_height}")

        # Load target image
        target_img = cv2.imread(target_image_path)
        if target_img is None:
            print(f"Error: Could not load target image from {target_image_path}")
            return False

        target_height, target_width = target_img.shape[:2]
        print(f"Target image dimensions: {target_width} x {target_height}")

        # Compute homography between reference and target images
        H, img_reference, img_target = compute_homography_once(
            reference_image_path, target_image_path, matcher
        )

        if H is None:
            print("Error: Could not compute homography matrix between images")
            return False

        # Invert homography to warp target to reference frame
        H_inv = invert_homography(H)

        # Prepare reference image in BGR format
        R_prime_bgr = tensor_to_np(img_reference)
        # if R_prime_bgr.shape[2] == 3:
        #     R_prime_bgr = cv2.cvtColor(R_prime_bgr, cv2.COLOR_RGB2BGR)

        # Prepare target image
        img_target_np = tensor_to_np(img_target)

        # Add overlay text if requested
        if overlay_text:
            if text_to_overlay is None:
                text_to_overlay = os.path.splitext(os.path.basename(target_image_path))[
                    0
                ]
            img_target_np = add_text_to_image(img_target_np.copy(), text_to_overlay)

        # Warp target image to reference frame
        warped_img_np = warp_image(img_target_np, H_inv, (ref_width, ref_height))

        # Create mask for warping
        white_canvas = create_white_canvas_like(img_target_np)
        warped_mask = warp_image(white_canvas, H_inv, (ref_width, ref_height))

        # Combine reference and warped images
        combined_image = combine_images_with_masks(
            R_prime_bgr,
            warped_img_np,
            warped_mask,
            use_gradient_blending,
            blend_mode,
            feather_amount,
        )
        combined_image = normalize_and_convert_color(
            combined_image, convert=True
        )  # keep BGR for OpenCV

        # Ensure output has correct dimensions
        if combined_image.shape[:2] != (ref_height, ref_width):
            combined_image = cv2.resize(combined_image, (ref_width, ref_height))

        # Determine blending mode for display
        if use_gradient_blending and blend_mode == "edge_feather":
            display_mode = "poisson"
        else:
            display_mode = blend_mode

        # Save the warped image
        success = cv2.imwrite(output_image_path, combined_image)
        if success:
            print(
                f"Warped image saved to: {output_image_path} (using {display_mode} blending)"
            )
            return True
        else:
            print(f"Error: Could not save warped image to {output_image_path}")
            return False

    except Exception as e:
        print(f"Error in warp_image_to_reference: {str(e)}")
        return False


def warp_video_with_per_frame_homography(
    img_reference_path,
    target_video_path,
    matcher,
    parent_dir,
    output_video_path,
    overlay_text=True,
    use_gradient_blending=False,
    blend_mode="edge_feather",
    feather_amount=15,
):
    """
    Warp video by computing homography for each frame individually.

    Args:
        img_reference_path: Path to reference image
        target_video_path: Path to input video
        matcher: Feature matcher object
        parent_dir: Directory for temporary files
        output_video_path: Path for output video
        overlay_text: Whether to overlay filename on frames
        use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode)
        blend_mode: Blending mode - 'none', 'feathered', 'edge_feather', 'smart', 'multiband', or 'poisson'
        feather_amount: Feathering amount in pixels
    """
    file_name = os.path.splitext(os.path.basename(target_video_path))[0]
    ref_img = cv2.imread(img_reference_path)
    ref_height, ref_width = ref_img.shape[:2]
    print(f"Reference image dimensions: {ref_width} x {ref_height}")

    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Target video dimensions: {video_width} x {video_height}")

    # Use output_video_path instead of saving to parent_dir
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref_width, ref_height))

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Determine blending mode for display
        if use_gradient_blending and blend_mode == "feathered":
            display_mode = "poisson"
        else:
            display_mode = blend_mode

        with tqdm(
            total=total_frames,
            desc=f"Processing Video (per-frame H, {display_mode} blending)",
            unit="frame",
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Finished reading the video.")
                    break

                # Add file name text to the top left corner of the target frame (after warped, not combined)
                if overlay_text:
                    frame_with_text = add_text_to_image(frame.copy(), file_name)
                else:
                    frame_with_text = frame.copy()

                warped_frame = get_warped_frame_using_diff_H(
                    img_reference_path,
                    frame_with_text,
                    matcher,
                    parent_dir,
                    use_gradient_blending,
                    blend_mode,
                    feather_amount,
                )
                if warped_frame.shape[:2] != (ref_height, ref_width):
                    warped_frame = cv2.resize(warped_frame, (ref_width, ref_height))
                pbar.update(1)
                out.write(warped_frame)
            cv2.destroyAllWindows()
    cap.release()
    out.release()
