import cv2
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import os

from .kamlan_filter import SimpleKalmanFilter
from .file_operations import get_output_path
from .draw_helpers import (
    draw_trajectory,
    draw_velocity_arrow,
    draw_gauge,
    draw_label,
    GaugeConfig,
    get_gauge_centers,
)
from .pose_helpers import get_track_point_coords
from .pose_backend import PoseDetector, NormalizedPoseLandmark, draw_pose_landmarks

# Utils
from .utils.images import save_trajectories_as_png

trajectory_colors = {
    "matisse": {
        # in BGR format
        "red": (59, 67, 183),
        "green": (127, 161, 85),
        "blue": (192, 112, 78),
        "orange": (63, 125, 217),
        "yellow": (70, 181, 223),
        "magenta": (131, 78, 176),
        "purple": (188, 125, 160),
        "beige": (201, 218, 217),
    }
}


colors = {
    # B, G, R
    "hip_mid": trajectory_colors["matisse"]["red"],
    "upper_body_center": trajectory_colors["matisse"]["green"],
    "head": trajectory_colors["matisse"]["blue"],
    "left_hand": trajectory_colors["matisse"]["orange"],
    "right_hand": trajectory_colors["matisse"]["yellow"],
    "left_foot": trajectory_colors["matisse"]["magenta"],
    "right_foot": trajectory_colors["matisse"]["purple"],
}


def extract_pose_and_draw_trajectory(
    video_path,
    output_path=None,  # optional, if not provided, the output video will be saved in the `output` folder
    track_point=["hip_mid"],  # a list of track points to draw trajectory for
    hide_original_video=False,  # if True, the output video will have a black background instead of the original frames
    overlay_mask=False,  # if `True`, we draw trajectory on a semi-transparent black overlay
    overlay_trajectory=None,  # deprecated alias
    overlay_opacity=0.8,  # opacity for the overlay, value should between [0.0, 1.0]
    show_gauges=False,  # whether to show gauges and related text
    draw_pose=True,  # whether to draw the body pose skeleton
    pose_color=(
        255,
        255,
        255,
    ),  # Color for pose skeleton in BGR format (default: white)
    show_trajectory=True,  # whether to draw the trajectories
    kalman_settings=[True, 1e-1],  # [use_kalman, measurement_variance]
    trajectory_png_path=None,  # NEW: optional PNG output path
    savgol_settings=[False, 11, 3],  # [use_savgol, window_length, polyorder]
):
    # Suppress MediaPipe warnings
    os.environ["GLOG_minloglevel"] = "2"

    if overlay_trajectory is not None:
        overlay_mask = overlay_trajectory

    use_kalman = kalman_settings[0]  # whether to use Kalman filter
    measurement_variance = kalman_settings[1]  # variance for the Kalman filter

    # use_savgol = savgol_settings[0]  # whether to use Savitzky-Golay filter
    # Always use savgol

    savgol_window = savgol_settings[1]  # window length for Savgol filter (must be odd)
    savgol_order = savgol_settings[2]  # polynomial order for Savgol filter

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Store trajectories and 3D trajectories for each track point
    trajectories = {tp: [] for tp in track_point}
    trajectories_3d = {tp: [] for tp in track_point}
    max_observed_velocity = {tp: 0 for tp in track_point}

    # Initialize Kalman filters for each track point if enabled
    kalman_filters = (
        # measurement_variance=1e0 for high noise, 1e-2 for low noise
        # default is 1e-1
        {
            tp: SimpleKalmanFilter(measurement_variance=measurement_variance)
            for tp in track_point
        }
        if use_kalman
        else None
    )

    # Get video properties
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    effective_fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_detector = PoseDetector()

    # Set output path if not provided
    output_path = get_output_path(
        video_path,
        output_path,
        output_prefix="pose_trajectory",
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc if fourcc != 0 else cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Initialize overlay canvas if needed
    overlay_canvas = None

    # Initialize gauge configuration
    gauge_config = GaugeConfig()
    gauge_centers = get_gauge_centers(track_point, gauge_config)

    # If using Savgol filter, we need a two-pass approach:
    # Pass 1: Collect all raw landmarks and pose data
    # Pass 2: Apply filter to pose skeleton only (not trajectories) and render video

    # First pass: collect all landmarks and store pose landmarks for smoothing
    print("Savgol filter enabled: First pass - collecting landmarks...")
    frames_data = []  # Store frame data for second pass
    all_pose_landmarks = []  # Store all pose landmarks for smoothing

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        with tqdm(
            total=total_frames, desc="Collecting landmarks", unit="frame"
        ) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_data.append(frame.copy())
                timestamp_ms = int(len(frames_data) * 1000 / effective_fps)
                results = pose_detector.process(frame, timestamp_ms=timestamp_ms)

                all_pose_landmarks.append(results.pose_landmarks)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks
                    h, w, _ = frame.shape

                    for tp in track_point:
                        confidence_threshold = 0.6
                        result = get_track_point_coords(
                            tp, landmarks, w, h, confidence_threshold
                        )
                        if result is None:
                            trajectories[tp].append(None)
                            trajectories_3d[tp].append(None)
                        else:
                            mid_point, mid_point_3d = result
                            if use_kalman:
                                smoothed_mid_point = kalman_filters[tp].update(
                                    mid_point
                                )
                            else:
                                smoothed_mid_point = mid_point
                            trajectories[tp].append(smoothed_mid_point)
                            trajectories_3d[tp].append(mid_point_3d)
                else:
                    for tp in track_point:
                        trajectories[tp].append(None)
                        trajectories_3d[tp].append(None)

                pbar.update(1)

        print(
            f"Applying Savgol filter to pose skeleton (window={savgol_window}, order={savgol_order})..."
        )
        smoothed_pose_landmarks = []

        num_landmarks = 33
        for lm_idx in range(num_landmarks):
            valid_frames = []
            x_coords = []
            y_coords = []
            z_coords = []

            for frame_idx, pose_lm in enumerate(all_pose_landmarks):
                if pose_lm is not None:
                    valid_frames.append(frame_idx)
                    x_coords.append(pose_lm[lm_idx].x)
                    y_coords.append(pose_lm[lm_idx].y)
                    z_coords.append(pose_lm[lm_idx].z)

            if len(valid_frames) >= savgol_window:
                x_smooth = savgol_filter(x_coords, savgol_window, savgol_order)
                y_smooth = savgol_filter(y_coords, savgol_window, savgol_order)
                z_smooth = savgol_filter(z_coords, savgol_window, savgol_order)

                for idx, frame_idx in enumerate(valid_frames):
                    if frame_idx >= len(smoothed_pose_landmarks):
                        while len(smoothed_pose_landmarks) <= frame_idx:
                            smoothed_pose_landmarks.append({})

                    smoothed_pose_landmarks[frame_idx][lm_idx] = {
                        "x": x_smooth[idx],
                        "y": y_smooth[idx],
                        "z": z_smooth[idx],
                        "visibility": all_pose_landmarks[frame_idx][lm_idx].visibility,
                        "presence": all_pose_landmarks[frame_idx][lm_idx].presence,
                    }

        print(
            "Second pass - rendering video with raw trajectories and smoothed skeleton..."
        )
        frame_idx = 0

        with tqdm(total=len(frames_data), desc="Rendering video", unit="frame") as pbar:
            for frame in frames_data:
                if hide_original_video:
                    frame = np.zeros_like(frame)

                if overlay_mask:
                    if overlay_canvas is None:
                        overlay_canvas = np.zeros_like(frame)
                        overlay_canvas[:] = (0, 0, 0)

                smoothed_landmarks_for_drawing = None
                if (
                    draw_pose
                    and frame_idx < len(smoothed_pose_landmarks)
                    and smoothed_pose_landmarks[frame_idx]
                    and all_pose_landmarks[frame_idx]
                ):
                    smoothed_landmarks_for_drawing = [
                        NormalizedPoseLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            visibility=landmark.visibility,
                            presence=landmark.presence,
                        )
                        for landmark in all_pose_landmarks[frame_idx]
                    ]
                    for lm_idx in range(num_landmarks):
                        if lm_idx in smoothed_pose_landmarks[frame_idx]:
                            lm_data = smoothed_pose_landmarks[frame_idx][lm_idx]
                            smoothed_landmarks_for_drawing[lm_idx].x = lm_data["x"]
                            smoothed_landmarks_for_drawing[lm_idx].y = lm_data["y"]
                            smoothed_landmarks_for_drawing[lm_idx].z = lm_data["z"]
                            smoothed_landmarks_for_drawing[lm_idx].visibility = lm_data[
                                "visibility"
                            ]
                            smoothed_landmarks_for_drawing[lm_idx].presence = lm_data[
                                "presence"
                            ]

                # Draw trajectories up to current frame (using raw, unsmoothed data)
                for idx, tp in enumerate(track_point):
                    # Get trajectory up to current frame
                    traj = [
                        p for p in trajectories[tp][: frame_idx + 1] if p is not None
                    ]
                    traj_3d = [
                        p for p in trajectories_3d[tp][: frame_idx + 1] if p is not None
                    ]
                    color = colors.get(tp, (0, 255, 255))

                    # Draw trajectory if enabled
                    if show_trajectory:
                        if overlay_mask:
                            draw_trajectory(overlay_canvas, traj, color, thickness=2)
                        else:
                            draw_trajectory(frame, traj, color, thickness=2)

                    # Draw velocity and gauges
                    if len(traj) > 1 and len(traj_3d) > 1:
                        prev_point = traj[-2]
                        curr_point = traj[-1]
                        # Draw velocity arrow only if showing trajectories and not using overlay
                        if show_trajectory and not overlay_mask:
                            draw_velocity_arrow(
                                frame,
                                prev_point,
                                curr_point,
                                color,
                                scale=5,
                                thickness=3,
                            )

                        prev_3d = traj_3d[-2]
                        curr_3d = traj_3d[-1]
                        velocity_3d = (
                            curr_3d[0] - prev_3d[0],
                            curr_3d[1] - prev_3d[1],
                            curr_3d[2] - prev_3d[2],
                        )
                        abs_velocity = (
                            velocity_3d[0] ** 2
                            + velocity_3d[1] ** 2
                            + velocity_3d[2] ** 2
                        ) ** 0.5
                        abs_velocity *= 1000

                        if abs_velocity > max_observed_velocity[tp]:
                            max_observed_velocity[tp] = abs_velocity

                        if show_gauges:
                            center = gauge_centers[idx]
                            max_velocity = gauge_config.max_velocity
                            velocity_clamped = min(abs_velocity, max_velocity)
                            angle = int((velocity_clamped / max_velocity) * 270)
                            start_angle = 135
                            end_angle = start_angle + angle
                            gauge_color = (
                                int(0 + 255 * (velocity_clamped / max_velocity)),
                                int(255 - 255 * (velocity_clamped / max_velocity)),
                                0,
                            )
                            gauge_canvas = overlay_canvas if overlay_mask else frame
                            velocity_text = f"{abs_velocity:.1f} mm/frame"
                            max_velocity_text = (
                                f"Max: {max_observed_velocity[tp]:.1f} mm/frame"
                            )
                            draw_gauge(
                                gauge_canvas,
                                center,
                                gauge_config.radius,
                                gauge_config.thickness,
                                start_angle,
                                end_angle,
                                gauge_color,
                                max_velocity,
                                abs_velocity,
                                velocity_text,
                                max_velocity_text,
                            )
                            draw_label(
                                gauge_canvas,
                                center,
                                gauge_config.radius,
                                tp,
                                color,
                            )

                if overlay_mask and overlay_canvas is not None:
                    blended = cv2.addWeighted(
                        frame, 1 - overlay_opacity, overlay_canvas, overlay_opacity, 0
                    )
                    frame = blended

                if draw_pose and smoothed_landmarks_for_drawing:
                    draw_pose_landmarks(
                        frame,
                        smoothed_landmarks_for_drawing,
                        color=pose_color,
                        thickness=2,
                    )

                out.write(frame)
                frame_idx += 1
                pbar.update(1)
    finally:
        pose_detector.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Save PNG with just the trajectories if requested
    if trajectory_png_path is not None:
        # from utils.body_trajectory import save_trajectories_as_png
        save_trajectories_as_png(
            trajectories, width, height, trajectory_png_path, colors=colors
        )
