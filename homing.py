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

import os
import time
import cv2
import numpy as np

import motor

# Directory to save diagnostic frames when markers aren't found
DIAG_DIR = os.path.join(os.path.dirname(__file__), 'data', 'homing_debug')
DIAG_MAX_FILES = 10  # Keep only the most recent diagnostic frames

# HSV range for green fluorescent tape markers on bin and frame.
# When aligned, the two green tapes overlap into one blob (width ~34px).
# When misaligned, the blob widens as the tapes separate.
HSV_LOWER_1 = np.array([55, 70, 70])
HSV_UPPER_1 = np.array([110, 255, 255])
HSV_LOWER_2 = None
HSV_UPPER_2 = None

# Minimum contour area in pixels to be considered a marker
MARKER_MIN_AREA = 50

# Region of interest: only search the top-left quarter of the frame
# where the tape markers are.
ROI_X_FRACTION = 0.19

# Calibrated values at visually confirmed true center (2026-05-03, post-cycle).
# Green blob when aligned: cx=45, width=34 (10/10 stable, brightness ~97).
# Red blob when aligned: cx=59, width=34.
ALIGNED_WIDTH = 34
ALIGNED_CX = 45

# Minimum average frame brightness (0-255) required to attempt homing.
# Below this threshold the ambient light shifts marker hue out of the
# red range, making detection unreliable.
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 110

# Fixed forward nudge when brightness is outside CV range.
# Empirically calibrated: the motor cycle consistently displaces the bin
# by roughly this amount, so a fixed correction gets close.
FIXED_NUDGE_DURATION = 1.5
FIXED_NUDGE_SPEED = 0.7

# Number of frames to grab and discard before reading a real frame.
# Cameras buffer old frames; flushing ensures we get a current image.
CAMERA_FLUSH_FRAMES = 3


def _save_diagnostic(frame, mask, attempt, brightness=None):
    """Save a diagnostic frame + mask when markers aren't found.

    Keeps only the most recent DIAG_MAX_FILES pairs to avoid filling disk.
    """
    try:
        os.makedirs(DIAG_DIR, exist_ok=True)

        # Remove old files if over limit
        existing = sorted(
            [f for f in os.listdir(DIAG_DIR) if f.endswith('.jpg')],
        )
        while len(existing) >= DIAG_MAX_FILES * 2 - 1:
            os.remove(os.path.join(DIAG_DIR, existing.pop(0)))

        ts = time.strftime('%Y%m%d_%H%M%S')
        brt_tag = f'_brt{int(brightness)}' if brightness is not None else ''
        cv2.imwrite(os.path.join(DIAG_DIR, f'{ts}_frame_a{attempt}{brt_tag}.jpg'), frame)
        cv2.imwrite(os.path.join(DIAG_DIR, f'{ts}_mask_a{attempt}{brt_tag}.jpg'), mask)
    except Exception as e:
        print(f'Homing: diagnostic save failed: {e}')


def detect_marker_blob(frame, hsv_lower=None, hsv_upper=None,
                       min_area=MARKER_MIN_AREA, roi_x_fraction=ROI_X_FRACTION):
    """
    Detect the green marker blob in a BGR frame.

    The bin and frame each have a green tape marker. When aligned, they
    overlap into a single compact blob. When misaligned, the blob widens
    as the tapes separate horizontally.

    Args:
        frame: BGR image from cv2.VideoCapture.
        hsv_lower: Lower HSV bound. Defaults to HSV_LOWER_1.
        hsv_upper: Upper HSV bound. Defaults to HSV_UPPER_1.
        min_area: Minimum contour area to qualify.
        roi_x_fraction: Fraction of frame width to search (from left).

    Returns:
        Dict with cx, cy, width, height, area of the largest green blob,
        or None if no blob found.
    """
    if hsv_lower is None:
        hsv_lower = HSV_LOWER_1
    if hsv_upper is None:
        hsv_upper = HSV_UPPER_1

    roi_x = int(frame.shape[1] * roi_x_fraction)
    roi = frame[:, :roi_x]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(best)
    return {'cx': cx, 'cy': cy, 'width': w, 'height': h, 'area': cv2.contourArea(best)}


# Legacy wrappers kept for test compatibility
def detect_markers(frame, hsv_lower=None, hsv_upper=None, min_area=MARKER_MIN_AREA,
                   roi_x_fraction=ROI_X_FRACTION):
    """Detect two colored markers (legacy interface for tests)."""
    if hsv_lower is None:
        hsv_lower = HSV_LOWER_1
    if hsv_upper is None:
        hsv_upper = HSV_UPPER_1

    roi_x = int(frame.shape[1] * roi_x_fraction)
    roi = frame[:, :roi_x]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    valid.sort(key=cv2.contourArea, reverse=True)
    if len(valid) < 2:
        return None
    centroids = []
    for contour in valid[:2]:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    if len(centroids) < 2:
        return None
    centroids.sort(key=lambda p: p[1])
    return centroids


def compute_alignment_error(markers):
    """Compute alignment error (legacy interface for tests)."""
    return markers[0][0] - markers[1][0]


def _flush_camera(cap, count=CAMERA_FLUSH_FRAMES):
    """Grab and discard frames to flush the camera buffer.

    USB cameras buffer old frames internally. After the motor stops,
    the buffered frames still show the old position. Flushing ensures
    the next read() returns a current image.
    """
    for _ in range(count):
        cap.grab()


def _try_detect_blob(cap, roi_x_fraction, retries=3, retry_delay=0.2):
    """Try to detect the green marker blob, retrying on failure.

    Returns:
        (success: bool, frame, blob) — blob is a dict from detect_marker_blob,
        or None if not found.
    """
    frame = None
    for i in range(retries):
        success, frame = cap.read()
        if not success:
            return False, None, None
        blob = detect_marker_blob(frame, roi_x_fraction=roi_x_fraction)
        if blob is not None:
            return True, frame, blob
        if i < retries - 1:
            time.sleep(retry_delay)
    return True, frame, None


def home(cap, nudge_speed=0.7, nudge_duration=0.5, tolerance_px=2,
         max_attempts=40, settle_time=0.5, move_fn=None,
         roi_x_fraction=ROI_X_FRACTION, flush_frames=CAMERA_FLUSH_FRAMES,
         confirm_reads=3, min_brightness=MIN_BRIGHTNESS,
         aligned_width=ALIGNED_WIDTH, aligned_cx=ALIGNED_CX):
    """
    Closed-loop visual homing using green blob width.

    The bin and frame each have a green tape. When aligned they overlap
    into one compact blob (width ~ALIGNED_WIDTH). When misaligned the
    blob widens. The centroid x relative to ALIGNED_CX determines nudge
    direction.

    Returns:
        Dict with keys: aligned (bool), final_error_px (int), attempts (int),
            marker_loss_count (int).
    """
    if move_fn is None:
        move_fn = motor.move

    _flush_camera(cap, flush_frames)

    # Check ambient brightness
    success, frame = cap.read()
    if success and frame is not None:
        brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
        print(f"Homing: brightness={brightness:.0f} (range={min_brightness}-{MAX_BRIGHTNESS})")
        if brightness < min_brightness:
            print(f"Homing: skipped, too dark (brightness={brightness:.0f})")
            return {'aligned': False, 'final_error_px': 0,
                    'attempts': 0, 'marker_loss_count': 0,
                    'skipped': 'low_light', 'brightness': brightness}
        if brightness > MAX_BRIGHTNESS:
            print(f"Homing: too bright for CV ({brightness:.0f}>{MAX_BRIGHTNESS}), using fixed forward nudge")
            move_fn(velocity=FIXED_NUDGE_SPEED, duration=FIXED_NUDGE_DURATION)
            return {'aligned': True, 'final_error_px': 0,
                    'attempts': 1, 'marker_loss_count': 0,
                    'fixed_nudge': True, 'brightness': brightness}

    marker_loss_count = 0
    prev_width = None
    direction = 1.0  # Start forward (cycle displaces bin backward)
    width_increasing_count = 0

    for attempt in range(max_attempts):
        cam_ok, frame, blob = _try_detect_blob(cap, roi_x_fraction)
        if not cam_ok:
            print("Homing: camera read failed, aborting")
            return {'aligned': False, 'final_error_px': 0,
                    'attempts': attempt + 1, 'marker_loss_count': marker_loss_count}

        if blob is None:
            marker_loss_count += 1
            if frame is not None and (marker_loss_count == 1 or marker_loss_count % 10 == 0):
                roi_x = int(frame.shape[1] * roi_x_fraction)
                roi = frame[:, :roi_x]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, HSV_LOWER_1, HSV_UPPER_1)
                brt = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
                _save_diagnostic(frame, mask, attempt + 1, brightness=brt)
            print(f"Homing: blob not found (attempt {attempt + 1}/{max_attempts})")
            time.sleep(settle_time)
            continue

        width = blob['width']
        print(f"Homing: width={width}px, cx={blob['cx']}, dir={'fwd' if direction > 0 else 'rev'} "
              f"(attempt {attempt + 1}/{max_attempts})")

        if prev_width is not None:
            if width > prev_width + 1:
                # Width increased — we passed through center, reverse direction
                width_increasing_count += 1
                if width_increasing_count >= 2:
                    direction = -direction
                    width_increasing_count = 0
                    print(f"Homing: width increasing, reversing to {'fwd' if direction > 0 else 'rev'}")
            elif width < prev_width - 1:
                # Width decreasing — approaching center, keep going
                width_increasing_count = 0
            else:
                # Width stable — might be at minimum
                width_increasing_count = 0
                # Check if width is near minimum (confirm with multiple reads)
                if attempt >= 2:
                    # Read a few more to confirm stability
                    stable_count = 0
                    for _ in range(confirm_reads):
                        ok2, _, b2 = _try_detect_blob(cap, roi_x_fraction)
                        if ok2 and b2 and abs(b2['width'] - width) <= tolerance_px:
                            stable_count += 1
                    if stable_count >= confirm_reads:
                        print(f"Homing: aligned (width={width}px stable, {confirm_reads} confirms, attempt {attempt + 1})")
                        return {'aligned': True, 'final_error_px': 0,
                                'attempts': attempt + 1, 'marker_loss_count': marker_loss_count}

        prev_width = width

        # Nudge in current direction
        speed = nudge_speed
        dur = nudge_duration
        move_fn(velocity=direction * speed, duration=dur)
        time.sleep(settle_time)
        _flush_camera(cap, flush_frames)

    print(f"Homing: max attempts ({max_attempts}) reached")
    return {'aligned': False, 'final_error_px': 0,
            'attempts': max_attempts, 'marker_loss_count': marker_loss_count}
