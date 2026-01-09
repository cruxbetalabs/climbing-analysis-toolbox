import cv2
import numpy as np

def save_trajectories_as_png(
    trajectories, width, height, output_path, colors=None, thickness=2
):
    """
    Save a PNG image with just the trajectories drawn.
    Args:
        trajectories: dict of {track_point: list of (x, y) tuples}
        width: image width
        height: image height
        output_path: path to save the PNG
        colors: dict of {track_point: (B, G, R)}
        thickness: line thickness
    """
    # Create a blank black image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if colors is None:
        colors = {tp: (0, 255, 255) for tp in trajectories}
    for tp, traj in trajectories.items():
        color = colors.get(tp, (0, 255, 255))
        for i in range(1, len(traj)):
            cv2.line(img, traj[i - 1], traj[i], color, thickness)
    cv2.imwrite(output_path, img)