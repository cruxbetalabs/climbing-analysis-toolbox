import cv2
import numpy as np


def draw_trajectory(canvas, traj, color, thickness=2):
    for i in range(1, len(traj)):
        cv2.line(canvas, traj[i - 1], traj[i], color, thickness)


def draw_velocity_arrow(canvas, prev_point, curr_point, color, scale=5, thickness=3):
    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    direction_norm = np.hypot(dx, dy)
    if direction_norm == 0:
        return

    arrow_length = scale
    direction_x = dx / direction_norm
    direction_y = dy / direction_norm
    end_point = (
        curr_point[0] + int(direction_x * arrow_length),
        curr_point[1] + int(direction_y * arrow_length),
    )
    cv2.arrowedLine(canvas, curr_point, end_point, color, thickness, tipLength=0.3)


def draw_telemetry_panel(canvas, telemetry_rows, origin=(20, 20)):
    if not telemetry_rows:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    line_height = 24
    header = "joint | ref_v | vel_ratio | arrow_len"
    padding_x = 10
    padding_y = 10

    rendered_rows = [header] + telemetry_rows
    text_width = max(
        cv2.getTextSize(row, font, font_scale, text_thickness)[0][0]
        for row in rendered_rows
    )
    panel_width = text_width + padding_x * 2
    panel_height = padding_y * 2 + line_height * len(rendered_rows)
    x0, y0 = origin
    x1 = x0 + panel_width
    y1 = y0 + panel_height

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (230, 230, 230), 1)

    for row_idx, row in enumerate(rendered_rows):
        text_y = y0 + padding_y + (row_idx + 1) * line_height - 6
        color = (230, 230, 230) if row_idx == 0 else (245, 245, 245)
        cv2.putText(
            canvas,
            row,
            (x0 + padding_x, text_y),
            font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
