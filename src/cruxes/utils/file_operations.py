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
