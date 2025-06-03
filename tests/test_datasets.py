import numpy as np
import pytest
from utils import datasets

class DummyCapture:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.opened_calls = 0
    def isOpened(self):
        self.opened_calls += 1
        return self.opened_calls == 1
    def get(self, prop):
        if prop == datasets.cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == datasets.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == datasets.cv2.CAP_PROP_FPS:
            return 0
        return 0
    def read(self):
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)
    def grab(self):
        pass
    def retrieve(self):
        return self.read()

class DummyThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args
    def start(self):
        pass

def test_loadstreams_defaults_fps(monkeypatch):
    dummy_cap = DummyCapture()
    monkeypatch.setattr(datasets.cv2, 'VideoCapture', lambda x: dummy_cap)
    monkeypatch.setattr(datasets, 'Thread', DummyThread)
    monkeypatch.setattr(datasets, 'letterbox', lambda img, *a, **k: (img, (1, 1), (0, 0)))
    monkeypatch.setattr(datasets.time, 'sleep', lambda x: None)

    loader = datasets.LoadStreams('dummy.mp4')
    assert loader.fps == 30

    # Should not raise ZeroDivisionError even though CAP_PROP_FPS returned 0
    loader.update(0, dummy_cap)
