import numpy as np
from utils.frame_utils import is_duplicate_frame


def test_identical_frames_are_duplicates():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert is_duplicate_frame(frame, frame.copy()) is True


def test_different_frames_are_not_duplicates():
    frame_a = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_b = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    assert is_duplicate_frame(frame_a, frame_b) is False


def test_none_last_frame_is_not_duplicate():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert is_duplicate_frame(frame, None) is False
