# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Visual homing module for OpenLitterPI.

Uses two colored markers (one on the bin, one on the frame) to detect
alignment. After the timed motor cycle, nudges the motor until the
markers are horizontally aligned, correcting cumulative drift.
"""

import time
import cv2
import numpy as np

import motor

# Default HSV range for fluorescent green marker.
# Tune these for your specific marker color and lighting.
HSV_LOWER = np.array([35, 80, 80])
HSV_UPPER = np.array([85, 255, 255])

# Minimum contour area in pixels to be considered a marker
MARKER_MIN_AREA = 30

# Region of interest: only search the left portion of the frame
# where the markers are, ignoring window/trees on the right.
# Value is a fraction of frame width (0.0 to 1.0).
ROI_X_FRACTION = 0.35

# Known x-offset between markers when bin is physically aligned.
# Measured as compute_alignment_error() when bin is in correct position.
ALIGNED_OFFSET = 31


def detect_markers(frame, hsv_lower=None, hsv_upper=None, min_area=MARKER_MIN_AREA,
                   roi_x_fraction=ROI_X_FRACTION):
    """
    Detect two colored markers in a BGR frame.

    Args:
        frame: BGR image from cv2.VideoCapture
        hsv_lower: Lower HSV bound (numpy array). Defaults to HSV_LOWER.
        hsv_upper: Upper HSV bound (numpy array). Defaults to HSV_UPPER.
        min_area: Minimum contour area to qualify as a marker.
        roi_x_fraction: Fraction of frame width to search (from left).
            Set to 1.0 to search the full frame.

    Returns:
        List of two (x, y) centroids sorted by y-coordinate (top first),
        or None if fewer than 2 markers found.
    """
    if hsv_lower is None:
        hsv_lower = HSV_LOWER
    if hsv_upper is None:
        hsv_upper = HSV_UPPER

    # Crop to left ROI to ignore background (window/trees)
    roi_x = int(frame.shape[1] * roi_x_fraction)
    roi = frame[:, :roi_x]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and sort by area descending to get the two largest
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    valid.sort(key=cv2.contourArea, reverse=True)

    if len(valid) < 2:
        return None

    centroids = []
    for contour in valid[:2]:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    if len(centroids) < 2:
        return None

    # Sort by y-coordinate (top marker first)
    centroids.sort(key=lambda p: p[1])
    return centroids


def compute_alignment_error(markers):
    """
    Compute horizontal alignment error between two markers.

    Args:
        markers: List of two (x, y) centroids.

    Returns:
        x-difference (markers[0].x - markers[1].x). Positive means
        the top marker is to the right of the bottom marker.
    """
    return markers[0][0] - markers[1][0]


def home(cap, nudge_speed=0.7, nudge_duration=0.3, tolerance_px=15,
         max_attempts=20, settle_time=0.3, move_fn=None,
         roi_x_fraction=ROI_X_FRACTION):
    """
    Closed-loop visual homing routine.

    Captures frames, detects marker alignment, and nudges the motor
    until the two markers are horizontally aligned.

    Args:
        cap: cv2.VideoCapture object (must be open).
        nudge_speed: Motor speed for correction nudges (0.0-1.0).
        nudge_duration: Duration of each nudge in seconds.
        tolerance_px: Alignment tolerance in pixels.
        max_attempts: Maximum correction attempts before giving up.
        settle_time: Seconds to wait after each nudge for motor to stop.
        move_fn: Motor move function (for testing). Defaults to motor.move.

    Returns:
        True if markers are aligned within tolerance, False otherwise.
    """
    if move_fn is None:
        move_fn = motor.move

    for attempt in range(max_attempts):
        success, frame = cap.read()
        if not success:
            print("Homing: camera read failed, aborting")
            return False

        markers = detect_markers(frame, roi_x_fraction=roi_x_fraction)
        if markers is None:
            print(f"Homing: markers not found (attempt {attempt + 1}/{max_attempts})")
            time.sleep(settle_time)
            continue

        raw_error = compute_alignment_error(markers)
        error = raw_error - ALIGNED_OFFSET
        print(f"Homing: error={error}px (raw={raw_error}px), markers={markers} (attempt {attempt + 1}/{max_attempts})")

        if abs(error) <= tolerance_px:
            print(f"Homing: aligned (error={error}px, attempt {attempt + 1})")
            return True

        # Nudge motor to correct: scale nudge strength by error magnitude.
        # Strong nudges when far off, gentle nudges when close.
        direction = -1.0 if error > 0 else 1.0
        if abs(error) > 80:
            speed = nudge_speed
            duration = nudge_duration
        elif abs(error) > 40:
            speed = nudge_speed * 0.8
            duration = nudge_duration * 0.8
        else:
            speed = nudge_speed * 0.6
            duration = nudge_duration * 0.6
        move_fn(velocity=direction * speed, duration=duration)
        time.sleep(settle_time)

    print(f"Homing: max attempts ({max_attempts}) reached, accepting position")
    return False
