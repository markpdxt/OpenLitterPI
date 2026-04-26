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
from homing import detect_markers, compute_alignment_error, home


# --- Helpers ---

def make_frame_with_markers(positions, color_bgr=(0, 255, 0), radius=15,
                            width=640, height=480):
    """Create a synthetic BGR frame with colored circles as markers."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for (x, y) in positions:
        cv2.circle(frame, (x, y), radius, color_bgr, -1)
    return frame


# HSV range for pure green circles on black background
GREEN_LOWER = np.array([35, 100, 100])
GREEN_UPPER = np.array([85, 255, 255])


# --- detect_markers tests ---

class TestDetectMarkers:
    def test_finds_two_aligned_markers(self):
        """Two green circles at the same x should be detected."""
        frame = make_frame_with_markers([(320, 100), (320, 380)])
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is not None
        assert len(markers) == 2
        # Top marker first (lower y)
        assert markers[0][1] < markers[1][1]

    def test_returns_none_with_zero_markers(self):
        """Empty frame should return None."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is None

    def test_returns_none_with_one_marker(self):
        """Single marker should return None (need two)."""
        frame = make_frame_with_markers([(320, 240)])
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is None

    def test_ignores_small_contours(self):
        """Markers smaller than min_area should be ignored."""
        frame = make_frame_with_markers([(320, 100), (320, 380)], radius=2)
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, min_area=500, roi_x_fraction=1.0)
        assert markers is None

    def test_centroid_accuracy(self):
        """Detected centroids should be close to the drawn positions."""
        frame = make_frame_with_markers([(200, 100), (400, 350)])
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is not None
        # Top marker (y=100)
        assert abs(markers[0][0] - 200) < 3
        assert abs(markers[0][1] - 100) < 3
        # Bottom marker (y=350)
        assert abs(markers[1][0] - 400) < 3
        assert abs(markers[1][1] - 350) < 3

    def test_selects_two_largest(self):
        """With three markers, the two largest should be selected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)  # large
        cv2.circle(frame, (300, 300), 20, (0, 255, 0), -1)  # large
        cv2.circle(frame, (500, 200), 5, (0, 255, 0), -1)   # small
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, min_area=50, roi_x_fraction=1.0)
        assert markers is not None
        assert len(markers) == 2

    def test_wrong_color_not_detected(self):
        """Red circles should not be detected with green HSV range."""
        frame = make_frame_with_markers([(320, 100), (320, 380)],
                                        color_bgr=(0, 0, 255))
        markers = detect_markers(frame, GREEN_LOWER, GREEN_UPPER, roi_x_fraction=1.0)
        assert markers is None


# --- compute_alignment_error tests ---

class TestComputeAlignmentError:
    def test_aligned_markers(self):
        """Same x-coordinate should give zero error."""
        error = compute_alignment_error([(320, 100), (320, 380)])
        assert error == 0

    def test_positive_error(self):
        """Top marker right of bottom -> positive error."""
        error = compute_alignment_error([(330, 100), (320, 380)])
        assert error == 10

    def test_negative_error(self):
        """Top marker left of bottom -> negative error."""
        error = compute_alignment_error([(310, 100), (320, 380)])
        assert error == -10


# --- home() integration tests ---

class FakeCapture:
    """Mock cv2.VideoCapture that returns pre-defined frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame


class TestHome:
    def test_already_aligned(self):
        """If markers are already aligned, home returns True immediately."""
        frame = make_frame_with_markers([(320, 100), (320, 380)])
        cap = FakeCapture([frame])
        moves = []
        result = home(cap, move_fn=lambda **kw: moves.append(kw),
                      settle_time=0, tolerance_px=10, roi_x_fraction=1.0)
        assert result is True
        assert len(moves) == 0

    def test_nudges_to_align(self):
        """Simulates marker moving closer to alignment after each nudge."""
        frames = [
            make_frame_with_markers([(350, 100), (320, 380)]),  # error=30
            make_frame_with_markers([(335, 100), (320, 380)]),  # error=15
            make_frame_with_markers([(325, 100), (320, 380)]),  # error=5 -> aligned
        ]
        cap = FakeCapture(frames)
        moves = []
        result = home(cap, move_fn=lambda **kw: moves.append(kw),
                      settle_time=0, tolerance_px=10, roi_x_fraction=1.0)
        assert result is True
        assert len(moves) == 2  # two nudges before alignment

    def test_max_attempts_exceeded(self):
        """If never aligned, home returns False after max_attempts."""
        frame = make_frame_with_markers([(400, 100), (320, 380)])  # error=80
        # Provide enough identical frames for all attempts
        cap = FakeCapture([frame] * 5)
        moves = []
        result = home(cap, move_fn=lambda **kw: moves.append(kw),
                      settle_time=0, max_attempts=5, tolerance_px=10,
                      roi_x_fraction=1.0)
        assert result is False
        assert len(moves) == 5

    def test_camera_failure(self):
        """If camera read fails, home returns False."""
        cap = FakeCapture([])
        result = home(cap, move_fn=lambda **kw: None, settle_time=0,
                      roi_x_fraction=1.0)
        assert result is False

    def test_markers_not_found(self):
        """If markers are not visible, home exhausts attempts."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = FakeCapture([blank] * 3)
        result = home(cap, move_fn=lambda **kw: None,
                      settle_time=0, max_attempts=3, roi_x_fraction=1.0)
        assert result is False

    def test_nudge_direction_positive_error(self):
        """Positive error (top marker right) should nudge with positive direction."""
        frames = [
            make_frame_with_markers([(350, 100), (320, 380)]),  # error=+30
            make_frame_with_markers([(320, 100), (320, 380)]),  # aligned
        ]
        cap = FakeCapture(frames)
        moves = []
        home(cap, move_fn=lambda **kw: moves.append(kw),
             settle_time=0, nudge_speed=0.4, roi_x_fraction=1.0)
        assert moves[0]['velocity'] > 0

    def test_nudge_direction_negative_error(self):
        """Negative error (top marker left) should nudge with negative direction."""
        frames = [
            make_frame_with_markers([(290, 100), (320, 380)]),  # error=-30
            make_frame_with_markers([(320, 100), (320, 380)]),  # aligned
        ]
        cap = FakeCapture(frames)
        moves = []
        home(cap, move_fn=lambda **kw: moves.append(kw),
             settle_time=0, nudge_speed=0.4, roi_x_fraction=1.0)
        assert moves[0]['velocity'] < 0
