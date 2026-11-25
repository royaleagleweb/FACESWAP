"""Temporal smoothing for landmarks, alignment, and masks.

A One Euro filter keeps slow motion steady and still follows a real head
turn. A scene cut or a large jump resets the filter so two different shots
are not blended together.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _smoothing_alpha(dt: float, cutoff: float) -> float:
    cutoff = max(float(cutoff), 1e-3)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(float(dt), 1e-3))


class OneEuroFilter:
    """Adaptive low-pass filter. The first sample is returned unchanged."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.prev: Optional[np.ndarray] = None
        self.prev_d: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None

    def reset(self) -> None:
        self.prev = None
        self.prev_d = None
        self.prev_t = None

    def __call__(self, time_s: float, value: np.ndarray) -> np.ndarray:
        sample = np.asarray(value, dtype=np.float64)
        if self.prev is None or self.prev.shape != sample.shape:
            self.prev = sample.copy()
            self.prev_d = np.zeros_like(sample)
            self.prev_t = float(time_s)
            return sample.copy()
        current = float(time_s)
        if self.prev_t is None or current <= self.prev_t:
            current = float(self.prev_t or 0.0) + (1.0 / 30.0)
        dt = current - float(self.prev_t)
        derivative = (sample - self.prev) / dt
        alpha_d = _smoothing_alpha(dt, self.d_cutoff)
        d_hat = alpha_d * derivative + (1.0 - alpha_d) * self.prev_d
        speed = float(np.mean(np.abs(d_hat)))
        alpha = _smoothing_alpha(dt, self.min_cutoff + self.beta * speed)
        filtered = alpha * sample + (1.0 - alpha) * self.prev
        self.prev = filtered
        self.prev_d = d_hat
        self.prev_t = current
        return filtered


def landmark_jumped(previous: Optional[np.ndarray], current: np.ndarray) -> bool:
    """True when the 5 points teleport, which is a scene cut or a bad detect."""
    if previous is None:
        return False
    prev = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    curr = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    if prev.shape != curr.shape or curr.shape[0] < 2:
        return True
    delta = float(np.linalg.norm(curr - prev, axis=1).mean())
    eye = float(np.linalg.norm(curr[1] - curr[0]))
    return delta > max(28.0, eye * 1.25)


def smooth_pair(previous: Optional[np.ndarray], current: np.ndarray, keep: float = 0.65) -> np.ndarray:
    """Exponential average. A shape change starts over."""
    current = np.asarray(current, dtype=np.float64)
    if previous is None or np.asarray(previous).shape != current.shape:
        return current.copy()
    mixed = keep * np.asarray(previous, dtype=np.float64) + (1.0 - keep) * current
    return mixed
