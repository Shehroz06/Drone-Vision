import numpy as np
from config.constants import FRAME_HASH_THRESHOLD


def is_duplicate_frame(frame, last_frame) -> bool:
    if last_frame is None:
        return False
    if frame.shape != last_frame.shape:
        return False
    diff = np.mean(np.abs(frame.astype(np.int16) - last_frame.astype(np.int16)))
    return diff < FRAME_HASH_THRESHOLD


def resize_frame(frame, width: int, height: int):
    import cv2
    return cv2.resize(frame, (width, height))


def bgr_to_rgb(frame):
    return frame[:, :, ::-1]
