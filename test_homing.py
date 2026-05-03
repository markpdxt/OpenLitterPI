# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the visual homing module."""

import pytest
import numpy as np
import cv2
from homing import (detect_markers, compute_alignment_error, detect_marker_blob,
                    home, ALIGNED_WIDTH, ALIGNED_CX)


# --- Helpers ---

def make_frame_with_markers(positions, color_bgr=(0, 255, 0), radius=15,
                            width=640, height=480):
    """Create a synthetic BGR frame with colored circles as markers."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for (x, y) in positions:
        cv2.circle(frame, (x, y), radius, color_bgr, -1)
    return frame


def make_green_blob(cx, cy, blob_width, blob_height=30, width=640, height=480):
    """Create a frame with a green rectangle of specific width at (cx, cy)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x1 = cx - blob_width // 2
    y1 = cy - blob_height // 2
    x2 = x1 + blob_width
    y2 = y1 + blob_height
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
    return frame


GREEN_LOWER = np.array([35, 100, 100])
GREEN_UPPER = np.array([85, 255, 255])


# --- detect_markers tests (legacy interface) ---

class TestDetectMarkers:
    def test_finds_two_aligned_markers(self):
        frame = make_frame_with_markers([(320, 100), (320, 380)], color_bgr=(0, 255, 0))
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is not None
        assert len(markers) == 2
        assert markers[0][1] < markers[1][1]

    def test_returns_none_with_zero_markers(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is None

    def test_returns_none_with_one_marker(self):
        frame = make_frame_with_markers([(320, 240)], color_bgr=(0, 255, 0))
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is None

    def test_ignores_small_contours(self):
        frame = make_frame_with_markers([(320, 100), (320, 380)], color_bgr=(0, 255, 0), radius=2)
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, min_area=200, roi_x_fraction=1.0)
        assert markers is None

    def test_centroid_accuracy(self):
        frame = make_frame_with_markers([(200, 100), (400, 350)], color_bgr=(0, 255, 0))
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is not None
        assert abs(markers[0][0] - 200) < 3
        assert abs(markers[0][1] - 100) < 3

    def test_selects_two_largest(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)
        cv2.circle(frame, (300, 300), 20, (0, 255, 0), -1)
        cv2.circle(frame, (500, 200), 5, (0, 255, 0), -1)
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, min_area=50, roi_x_fraction=1.0)
        assert markers is not None
        assert len(markers) == 2


# --- detect_marker_blob tests ---

class TestDetectMarkerBlob:
    def test_detects_green_blob(self):
        frame = make_green_blob(100, 100, 34)
        blob = detect_marker_blob(frame, roi_x_fraction=1.0)
        assert blob is not None
        assert abs(blob['cx'] - 100) < 3
        assert abs(blob['width'] - 34) < 3

    def test_returns_none_for_empty_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob = detect_marker_blob(frame, roi_x_fraction=1.0)
        assert blob is None

    def test_wider_blob_when_misaligned(self):
        narrow = make_green_blob(100, 100, 34)
        wide = make_green_blob(100, 100, 50)
        b_narrow = detect_marker_blob(narrow, roi_x_fraction=1.0)
        b_wide = detect_marker_blob(wide, roi_x_fraction=1.0)
        assert b_narrow['width'] < b_wide['width']

    def test_centroid_shifts_with_offset(self):
        left = make_green_blob(80, 100, 40)
        right = make_green_blob(120, 100, 40)
        b_left = detect_marker_blob(left, roi_x_fraction=1.0)
        b_right = detect_marker_blob(right, roi_x_fraction=1.0)
        assert b_left['cx'] < b_right['cx']


# --- compute_alignment_error tests (legacy) ---

class TestComputeAlignmentError:
    def test_aligned_markers(self):
        assert compute_alignment_error([(320, 100), (320, 380)]) == 0

    def test_positive_error(self):
        assert compute_alignment_error([(330, 100), (320, 380)]) == 10

    def test_negative_error(self):
        assert compute_alignment_error([(310, 100), (320, 380)]) == -10


# --- home() tests (gradient descent on width) ---

class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def grab(self):
        if self._index < len(self._frames):
            self._index += 1
            return True
        return False


class TestHome:
    def test_camera_failure(self):
        cap = FakeCapture([])
        result = home(cap, move_fn=lambda **kw: None, settle_time=0,
                      roi_x_fraction=1.0, flush_frames=0, min_brightness=0)
        assert result['aligned'] is False

    def test_blob_not_found(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = FakeCapture([blank] * 15)
        result = home(cap, move_fn=lambda **kw: None,
                      settle_time=0, max_attempts=3, roi_x_fraction=1.0,
                      flush_frames=0, min_brightness=0)
        assert result['aligned'] is False
        assert result['marker_loss_count'] == 3

    def test_starts_nudging_forward(self):
        """Gradient descent starts with forward nudges."""
        frames = [make_green_blob(100, 100, 50)] * 5
        cap = FakeCapture(frames)
        moves = []
        home(cap, move_fn=lambda **kw: moves.append(kw),
             settle_time=0, max_attempts=2, roi_x_fraction=1.0,
             flush_frames=0, min_brightness=0)
        assert len(moves) > 0
        assert moves[0]['velocity'] > 0  # starts forward

    def test_stable_width_declares_aligned(self):
        """If width is stable across reads, declares aligned."""
        frame = make_green_blob(100, 100, 34)
        # Need enough frames: 1 for brightness + attempts with retries + confirms
        cap = FakeCapture([frame] * 30)
        moves = []
        result = home(cap, move_fn=lambda **kw: moves.append(kw),
                      settle_time=0, tolerance_px=2, roi_x_fraction=1.0,
                      flush_frames=0, confirm_reads=3, min_brightness=0)
        assert result['aligned'] is True
