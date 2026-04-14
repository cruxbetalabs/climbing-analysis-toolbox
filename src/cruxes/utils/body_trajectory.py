import json
import cv2
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import os

from .kamlan_filter import SimpleKalmanFilter
from .file_operations import get_landmarks_json_path, get_output_path
from .draw_helpers import (
    draw_trajectory,
    draw_velocity_arrow,
    draw_telemetry_panel,
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


def _serialize_landmarks(landmarks):
    if landmarks is None:
        return None

    return [
        {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
            "presence": landmark.presence,
        }
        for landmark in landmarks
    ]


def _deserialize_landmarks(serialized_landmarks):
    if serialized_landmarks is None:
        return None

    return [
        NormalizedPoseLandmark(
            x=landmark["x"],
            y=landmark["y"],
            z=landmark["z"],
            visibility=landmark.get("visibility"),
            presence=landmark.get("presence"),
        )
        for landmark in serialized_landmarks
    ]


def _append_track_points(
    trajectories,
    trajectories_3d,
    track_point,
    landmarks,
    frame_shape,
    use_kalman,
    kalman_filters,
):
    if landmarks:
        h, w = frame_shape[:2]
        for tp in track_point:
            confidence_threshold = 0.6
            result = get_track_point_coords(tp, landmarks, w, h, confidence_threshold)
            if result is None:
                trajectories[tp].append(None)
                trajectories_3d[tp].append(None)
            else:
                mid_point, mid_point_3d = result
                if use_kalman:
                    smoothed_mid_point = kalman_filters[tp].update(mid_point)
                else:
                    smoothed_mid_point = mid_point
                trajectories[tp].append(smoothed_mid_point)
                trajectories_3d[tp].append(mid_point_3d)
        return

    for tp in track_point:
        trajectories[tp].append(None)
        trajectories_3d[tp].append(None)


def _build_landmarks_cache_metadata(
    video_path,
    total_frames,
    effective_fps,
    width,
    height,
):
    video_stats = os.stat(video_path)
    return {
        "cache_version": 1,
        "video": {
            "source_path": os.path.abspath(video_path),
            "file_name": os.path.basename(video_path),
            "file_size_bytes": video_stats.st_size,
            "file_mtime_ns": video_stats.st_mtime_ns,
            "frame_count": total_frames,
            "fps": effective_fps,
            "width": width,
            "height": height,
        },
    }


def _save_landmarks_cache(cache_path, metadata, all_pose_landmarks):
    cache_payload = {
        **metadata,
        "frames": [_serialize_landmarks(landmarks) for landmarks in all_pose_landmarks],
    }
    with open(cache_path, "w", encoding="utf-8") as file_obj:
        json.dump(cache_payload, file_obj)


def _load_landmarks_cache(
    cache_path,
    video_path,
    total_frames,
    effective_fps,
    width,
    height,
):
    try:
        with open(cache_path, "r", encoding="utf-8") as file_obj:
            cache_payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to read landmarks cache: {exc}"

    if not isinstance(cache_payload, dict) or "video" not in cache_payload:
        return None, "cache format is unsupported"

    video_metadata = cache_payload.get("video", {})
    frames_payload = cache_payload.get("frames")
    if not isinstance(frames_payload, list):
        return None, "cache does not contain frame landmarks"

    current_stats = os.stat(video_path)
    expected_metadata = {
        "source_path": os.path.abspath(video_path),
        "file_size_bytes": current_stats.st_size,
        "file_mtime_ns": current_stats.st_mtime_ns,
        "frame_count": total_frames,
        "width": width,
        "height": height,
    }

    for key, expected_value in expected_metadata.items():
        actual_value = video_metadata.get(key)
        if actual_value != expected_value:
            return None, f"cache metadata mismatch for {key}"

    cached_fps = video_metadata.get("fps")
    if cached_fps is None or abs(cached_fps - effective_fps) > 1e-6:
        return None, "cache metadata mismatch for fps"

    if len(frames_payload) != total_frames:
        return None, "cache frame count does not match the video"

    return [_deserialize_landmarks(frame_landmarks) for frame_landmarks in frames_payload], None


def extract_pose_and_draw_trajectory(
    video_path,
    output_path=None,  # optional, if not provided, the output video will be saved in the `output` folder
    track_point=["hip_mid"],  # a list of track points to draw trajectory for
    hide_original_video=False,  # if True, the output video will have a black background instead of the original frames
    overlay_mask=False,  # if `True`, we draw trajectory on a semi-transparent black overlay
    overlay_trajectory=None,  # deprecated alias
    overlay_opacity=0.8,  # opacity for the overlay, value should between [0.0, 1.0]
    show_gauges=False,  # whether to show top-left telemetry text
    draw_pose=True,  # whether to draw the body pose skeleton
    pose_color=(
        255,
        255,
        255,
    ),  # Color for pose skeleton in BGR format (default: white)
    show_trajectory=True,  # whether to draw the trajectories
    trajectory_history_seconds=None,  # if set, only show the last N seconds of the trajectory
    use_cached_landmarks=False,
    export_landmarks=False,
    landmarks_json_path=None,
    kalman_settings=[True, 1e-1],  # [use_kalman, measurement_variance]
    trajectory_png_path=None,  # NEW: optional PNG output path
    savgol_settings=[False, 11, 3],  # [use_savgol, window_length, polyorder]
):
    # Suppress MediaPipe warnings
    os.environ["GLOG_minloglevel"] = "2"

    if overlay_trajectory is not None:
        overlay_mask = overlay_trajectory

    landmarks_cache_path = None
    if use_cached_landmarks or export_landmarks:
        landmarks_cache_path = get_landmarks_json_path(
            video_path, landmarks_json_path=landmarks_json_path
        )

    use_kalman = kalman_settings[0]  # whether to use Kalman filter
    measurement_variance = kalman_settings[1]  # variance for the Kalman filter

    use_savgol = savgol_settings[0]  # whether to use Savitzky-Golay filter

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
    trajectory_history_frames = None
    if trajectory_history_seconds is not None:
        if trajectory_history_seconds <= 0:
            raise ValueError("trajectory_history_seconds must be greater than 0")
        trajectory_history_frames = max(
            1, int(round(trajectory_history_seconds * effective_fps))
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_detector = None

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

    reference_velocity_floor = 20.0

    # Use a two-pass approach so we can optionally smooth pose landmarks over time
    # while keeping trajectory extraction separate.

    # First pass: read frames and either collect or load pose landmarks.
    frames_data = []  # Store frame data for second pass
    all_pose_landmarks = []  # Store all pose landmarks for smoothing

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarks_cache_metadata = _build_landmarks_cache_metadata(
        video_path,
        total_frames,
        effective_fps,
        width,
        height,
    )

    cached_pose_landmarks = None
    if use_cached_landmarks and landmarks_cache_path and os.path.exists(landmarks_cache_path):
        cached_pose_landmarks, cache_error = _load_landmarks_cache(
            landmarks_cache_path,
            video_path,
            total_frames,
            effective_fps,
            width,
            height,
        )
        if cached_pose_landmarks is not None:
            print(f"Using cached landmarks from {landmarks_cache_path}")
        else:
            print(
                f"Landmarks cache at {landmarks_cache_path} could not be used: {cache_error}. Recomputing landmarks..."
            )
    elif use_cached_landmarks and landmarks_cache_path:
        print(
            f"Landmarks cache not found at {landmarks_cache_path}. Recomputing landmarks..."
        )

    first_pass_desc = "Collecting landmarks"
    if cached_pose_landmarks is not None:
        print("First pass - reading frames with cached landmarks...")
        first_pass_desc = "Loading cached landmarks"
    else:
        print("First pass - collecting landmarks...")

    try:
        if cached_pose_landmarks is None:
            pose_detector = PoseDetector()
        with tqdm(
            total=total_frames, desc=first_pass_desc, unit="frame"
        ) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_data.append(frame.copy())
                if cached_pose_landmarks is not None:
                    landmarks = cached_pose_landmarks[len(frames_data) - 1]
                else:
                    timestamp_ms = int(len(frames_data) * 1000 / effective_fps)
                    results = pose_detector.process(frame, timestamp_ms=timestamp_ms)
                    landmarks = results.pose_landmarks

                all_pose_landmarks.append(landmarks)
                _append_track_points(
                    trajectories,
                    trajectories_3d,
                    track_point,
                    landmarks,
                    frame.shape,
                    use_kalman,
                    kalman_filters,
                )

                pbar.update(1)

        if (
            cached_pose_landmarks is None
            and export_landmarks
            and landmarks_cache_path is not None
        ):
            _save_landmarks_cache(
                landmarks_cache_path,
                landmarks_cache_metadata,
                all_pose_landmarks,
            )
            print(f"Saved landmarks cache to {landmarks_cache_path}")

        smoothed_pose_landmarks = []

        num_landmarks = 33
        if use_savgol:
            print(
                f"Applying Savgol filter to pose skeleton (window={savgol_window}, order={savgol_order})..."
            )
            smoothed_pose_landmarks = [dict() for _ in all_pose_landmarks]

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
                        smoothed_pose_landmarks[frame_idx][lm_idx] = {
                            "x": x_smooth[idx],
                            "y": y_smooth[idx],
                            "z": z_smooth[idx],
                            "visibility": all_pose_landmarks[frame_idx][
                                lm_idx
                            ].visibility,
                            "presence": all_pose_landmarks[frame_idx][lm_idx].presence,
                        }

        print(
            "Second pass - rendering video with raw trajectories and "
            f"{'smoothed' if use_savgol else 'raw'} skeleton..."
        )
        frame_idx = 0

        with tqdm(total=len(frames_data), desc="Rendering video", unit="frame") as pbar:
            for frame in frames_data:
                if hide_original_video:
                    frame = np.zeros_like(frame)

                velocity_arrows = []
                telemetry_rows = []

                if overlay_mask:
                    if overlay_canvas is None or trajectory_history_frames is not None:
                        overlay_canvas = np.zeros_like(frame)
                        overlay_canvas[:] = (0, 0, 0)

                pose_landmarks_for_drawing = None
                if draw_pose and all_pose_landmarks[frame_idx]:
                    pose_landmarks_for_drawing = [
                        NormalizedPoseLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            visibility=landmark.visibility,
                            presence=landmark.presence,
                        )
                        for landmark in all_pose_landmarks[frame_idx]
                    ]
                    if use_savgol and frame_idx < len(smoothed_pose_landmarks):
                        for lm_idx in range(num_landmarks):
                            if lm_idx in smoothed_pose_landmarks[frame_idx]:
                                lm_data = smoothed_pose_landmarks[frame_idx][lm_idx]
                                pose_landmarks_for_drawing[lm_idx].x = lm_data["x"]
                                pose_landmarks_for_drawing[lm_idx].y = lm_data["y"]
                                pose_landmarks_for_drawing[lm_idx].z = lm_data["z"]
                                pose_landmarks_for_drawing[lm_idx].visibility = lm_data[
                                    "visibility"
                                ]
                                pose_landmarks_for_drawing[lm_idx].presence = lm_data[
                                    "presence"
                                ]

                # Draw trajectories up to the current frame using the collected track points.
                for idx, tp in enumerate(track_point):
                    history_start = 0
                    if trajectory_history_frames is not None:
                        history_start = max(
                            0, frame_idx + 1 - trajectory_history_frames
                        )

                    traj = [
                        p
                        for p in trajectories[tp][history_start : frame_idx + 1]
                        if p is not None
                    ]
                    traj_3d = [
                        p
                        for p in trajectories_3d[tp][history_start : frame_idx + 1]
                        if p is not None
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

                        reference_velocity = max(
                            reference_velocity_floor,
                            max_observed_velocity[tp],
                        )
                        velocity_ratio = min(
                            abs_velocity / reference_velocity,
                            1.0,
                        )
                        arrow_length = max(12, int(velocity_ratio * 60))

                        if show_trajectory:
                            velocity_arrows.append(
                                (
                                    prev_point,
                                    curr_point,
                                    color,
                                    arrow_length,
                                )
                            )

                        if show_gauges:
                            telemetry_rows.append(
                                f"{tp:<16} {reference_velocity:>6.1f} {velocity_ratio:>10.2f} {arrow_length:>10d}"
                            )
                    elif show_gauges:
                        telemetry_rows.append(
                            f"{tp:<16} {'--':>6} {'--':>10} {'--':>10}"
                        )

                if overlay_mask and overlay_canvas is not None:
                    blended = cv2.addWeighted(
                        frame, 1 - overlay_opacity, overlay_canvas, overlay_opacity, 0
                    )
                    frame = blended

                for prev_point, curr_point, color, arrow_length in velocity_arrows:
                    draw_velocity_arrow(
                        frame,
                        prev_point,
                        curr_point,
                        color,
                        scale=arrow_length,
                        thickness=3,
                    )

                if show_gauges:
                    draw_telemetry_panel(frame, telemetry_rows)

                if draw_pose and pose_landmarks_for_drawing:
                    draw_pose_landmarks(
                        frame,
                        pose_landmarks_for_drawing,
                        color=pose_color,
                        thickness=2,
                    )

                out.write(frame)
                frame_idx += 1
                pbar.update(1)
    finally:
        if pose_detector is not None:
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
