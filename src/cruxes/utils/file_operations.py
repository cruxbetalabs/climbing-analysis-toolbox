import os
import sys
from termcolor import colored


def get_output_path(input_video_path, output_video_path, output_prefix: str) -> str:
    if output_video_path is None:
        # store the output video in the same directory as the input video
        # extract the directory and file name from the video path
        input_dir = os.path.dirname(input_video_path)
        file_name = os.path.basename(input_video_path)

        # append a prefix to the file name
        file_name = f"{output_prefix}_{file_name}"

        # if input_dir is empty (file in current directory), use current directory
        if not input_dir:
            input_dir = "."

        output_path = os.path.join(input_dir, file_name)

        print(
            colored(
                f"Output video will be saved to {output_path}",
                "green",
                attrs=["bold"],
            )
        )

        return output_path
    else:
        # check if the output path specified is valid
        if not os.path.exists(os.path.dirname(output_video_path)):
            raise ValueError(
                f"Output path {output_video_path} does not exist. Please specify a valid path."
            )

        return output_video_path


def get_landmarks_json_path(input_video_path, landmarks_json_path=None) -> str:
    if landmarks_json_path is not None:
        output_dir = os.path.dirname(landmarks_json_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                f"Landmarks path {landmarks_json_path} does not exist. Please specify a valid path."
            )
        return landmarks_json_path

    input_dir = os.path.dirname(input_video_path) or "."
    file_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    return os.path.join(input_dir, f"{file_stem}_landmarks.json")


def get_trajectory_metadata_path(
    input_video_path, trajectory_metadata_path=None
) -> str:
    if trajectory_metadata_path is not None:
        output_dir = os.path.dirname(trajectory_metadata_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                f"Trajectory metadata path {trajectory_metadata_path} does not exist. Please specify a valid path."
            )
        return trajectory_metadata_path

    input_dir = os.path.dirname(input_video_path) or "."
    file_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    return os.path.join(input_dir, f"{file_stem}_trajectory_metadata.json")
