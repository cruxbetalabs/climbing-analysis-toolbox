import cv2
import numpy as np
from tqdm import tqdm
import os

# There's a package called `mediapipe-silicon` but it doesn't work with the latest version of `mediapipe`
import mediapipe as mp

from .file_operations import get_output_path


def segment_pose_from_video(
    video_path,
    output_path=None,  # optional, if not provided, the output video will be saved in the `output` folder
    background_color=(0, 255, 0),  # Background color in BGR format (default: green)
    segmentation_threshold=0.5,  # Threshold for segmentation confidence (0.0 to 1.0)
):
    """
    Segment the person from a video and replace the background with a solid color.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path for output video (auto-generated if None)
        background_color: Tuple of (B, G, R) values for background (default: green)
        segmentation_threshold: Confidence threshold for segmentation (default: 0.5)
    
    Returns:
        Path to the output video file
    """
    # Suppress MediaPipe warnings
    os.environ['GLOG_minloglevel'] = '2'
    
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    # Use model selection 1 for better quality (0 for general, 1 for landscape)
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set output path if not provided
    output_path = get_output_path(
        video_path,
        output_path,
        output_prefix="segmented",
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc if fourcc != 0 else cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Create background with specified color
    bg_image = np.zeros((height, width, 3), dtype=np.uint8)
    bg_image[:] = background_color

    print(f"Segmenting person from video with background color BGR{background_color}...")
    
    with tqdm(total=total_frames, desc="Segmenting frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to get segmentation mask
            results = segmentation.process(image_rgb)
            
            # Get the segmentation mask
            # The mask has values between 0 and 1, where values closer to 1 indicate person
            mask = results.segmentation_mask
            
            # Apply threshold to create binary mask
            binary_mask = (mask > segmentation_threshold).astype(np.uint8)
            
            # Create 3-channel mask for blending
            mask_3channel = np.stack([binary_mask] * 3, axis=-1)
            
            # Blend foreground (person) with background
            output_frame = np.where(mask_3channel, frame, bg_image)
            
            out.write(output_frame)
            pbar.update(1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Segmentation complete! Output saved to: {output_path}")
    return output_path
