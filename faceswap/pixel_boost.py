"""FaceFusion-style pixel boost for the 128px InSwapper.

A 256 or 512 aligned crop is split into 128px tiles. Each tile is swapped
and the tiles are stitched back, so the paste uses a sharper face than one
128px forward pass. 128 is a single tile, the same as a normal swap.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from .coverage import template_128

MODEL_SIZE = 128
PIXEL_BOOST_CHOICES = (128, 256, 512)


def normalize_pixel_boost(value: object) -> int:
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        size = MODEL_SIZE
    if size not in PIXEL_BOOST_CHOICES:
        return MODEL_SIZE
    return size


def pixel_boost_tiles(size: int) -> int:
    """How many InSwapper calls a boost setting makes. 256 is 4, 512 is 16."""
    size = normalize_pixel_boost(size)
    grid = size // MODEL_SIZE
    return grid * grid


def tile_slices(size: int) -> list[tuple[int, int, int, int]]:
    size = normalize_pixel_boost(size)
    grid = size // MODEL_SIZE
    slices = []
    for row in range(grid):
        for col in range(grid):
            y0 = row * MODEL_SIZE
            x0 = col * MODEL_SIZE
            slices.append((y0, x0, y0 + MODEL_SIZE, x0 + MODEL_SIZE))
    return slices


def split_tiles(crop: np.ndarray) -> list[np.ndarray]:
    size = int(crop.shape[0])
    return [crop[y0:y1, x0:x1] for y0, x0, y1, x1 in tile_slices(size)]


def join_tiles(tiles: list[np.ndarray], size: int) -> np.ndarray:
    size = normalize_pixel_boost(size)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for tile, (y0, x0, y1, x1) in zip(tiles, tile_slices(size)):
        patch = tile
        if patch.shape[0] != MODEL_SIZE or patch.shape[1] != MODEL_SIZE:
            patch = cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        canvas[y0:y1, x0:x1] = patch
    return canvas


def scaled_template(size: int) -> np.ndarray:
    return template_128() * (float(normalize_pixel_boost(size)) / float(MODEL_SIZE))


def frame_to_crop_matrix(kps: np.ndarray, size: int) -> Optional[np.ndarray]:
    """2×3 matrix that warps frame landmarks onto the scaled InSwapper template."""
    from .coverage import _estimate_similarity

    dst = scaled_template(size)
    src = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    return _estimate_similarity(src, dst)


def aligned_crop(frame: np.ndarray, kps: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    size = normalize_pixel_boost(size)
    matrix = frame_to_crop_matrix(kps, size)
    if matrix is None:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    crop = cv2.warpAffine(
        frame,
        matrix.astype(np.float32),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, np.asarray(matrix, dtype=np.float64)


def run_pixel_boost(
    frame: np.ndarray,
    kps: np.ndarray,
    size: int,
    swap_tile: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Swap each 128 tile of the aligned crop and return the stitched canvas.

    ``swap_tile`` receives one 128×128 BGR tile and returns the swapped tile.
    The matrix maps the original frame onto the returned canvas.
    """
    size = normalize_pixel_boost(size)
    crop, matrix = aligned_crop(frame, kps, size)
    if size == MODEL_SIZE:
        return swap_tile(crop), matrix
    swapped = [swap_tile(tile) for tile in split_tiles(crop)]
    return join_tiles(swapped, size), matrix
