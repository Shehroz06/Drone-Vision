import time
from collections import deque
from datetime import datetime
from config.constants import FPS_SAMPLE_WINDOW


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class FPSCounter:
    def __init__(self):
        self._timestamps = deque(maxlen=FPS_SAMPLE_WINDOW)

    def tick(self):
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
