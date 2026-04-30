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

# Default HSV range for red marker.
# Red wraps around H=0/180 in OpenCV, so we use two ranges and combine.
HSV_LOWER_1 = np.array([0, 100, 25])
HSV_UPPER_1 = np.array([12, 255, 255])
HSV_LOWER_2 = np.array([168, 100, 25])
HSV_UPPER_2 = np.array([180, 255, 255])

# Minimum contour area in pixels to be considered a marker
MARKER_MIN_AREA = 25

# Region of interest: only search the left portion of the frame
# where the markers are, ignoring window/trees on the right.
# Value is a fraction of frame width (0.0 to 1.0).
ROI_X_FRACTION = 0.35

# Minimum y-coordinate for valid marker contours.
# Filters out false positives near the top of the frame (e.g. reddish
# reflections on the litter box housing).
ROI_Y_MIN = 50

# Known x-offset between markers when bin is physically aligned.
# Measured as compute_alignment_error() when bin is in correct position.
# Red markers: roughly (59,70) and (44,82).
ALIGNED_OFFSET = 15

# Minimum average frame brightness (0-255) required to attempt homing.
# Below this threshold the ambient light shifts marker hue out of the
# red range, making detection unreliable.
MIN_BRIGHTNESS = 120

# Number of frames to grab and discard before reading a real frame.
# Cameras buffer old frames; flushing ensures we get a current image.
CAMERA_FLUSH_FRAMES = 3


def detect_markers(frame, hsv_lower=None, hsv_upper=None, min_area=MARKER_MIN_AREA,
                   roi_x_fraction=ROI_X_FRACTION):
    """
    Detect two colored markers in a BGR frame.

    Args:
        frame: BGR image from cv2.VideoCapture
        hsv_lower: Lower HSV bound (numpy array or list of arrays for multi-range).
            Defaults to red marker dual-range.
        hsv_upper: Upper HSV bound (numpy array or list of arrays for multi-range).
            Defaults to red marker dual-range.
        min_area: Minimum contour area to qualify as a marker.
        roi_x_fraction: Fraction of frame width to search (from left).
            Set to 1.0 to search the full frame.

    Returns:
        List of two (x, y) centroids sorted by y-coordinate (top first),
        or None if fewer than 2 markers found.
    """
    if hsv_lower is None:
        hsv_lower = [HSV_LOWER_1, HSV_LOWER_2]
        hsv_upper = [HSV_UPPER_1, HSV_UPPER_2]

    # Crop to left ROI to ignore background (window/trees)
    roi_x = int(frame.shape[1] * roi_x_fraction)
    roi = frame[:, :roi_x]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Support single range (legacy) or multi-range (red wraps around H=0/180)
    if isinstance(hsv_lower, list):
        mask = cv2.inRange(hsv, hsv_lower[0], hsv_upper[0])
        for lo, hi in zip(hsv_lower[1:], hsv_upper[1:]):
            mask = mask | cv2.inRange(hsv, lo, hi)
    else:
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and y-position, sort by area descending to get the two largest
    valid = [c for c in contours
             if cv2.contourArea(c) >= min_area
             and cv2.boundingRect(c)[1] >= ROI_Y_MIN]
    valid.sort(key=cv2.contourArea, reverse=True)

    if len(valid) < 2:
        # If only one large blob, try splitting it by finding the two
        # highest peaks in the y-projection (markers merged vertically)
        if len(valid) == 1 and cv2.contourArea(valid[0]) >= min_area * 4:
            x, y, w, h = cv2.boundingRect(valid[0])
            if h > w:  # vertically elongated = likely merged
                mid_y = y + h // 2
                top_mask = mask[y:mid_y, :]
                bot_mask = mask[mid_y:y+h, :]
                top_ctrs, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bot_ctrs, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if top_ctrs and bot_ctrs:
                    top_c = max(top_ctrs, key=cv2.contourArea)
                    bot_c = max(bot_ctrs, key=cv2.contourArea)
                    tM = cv2.moments(top_c)
                    bM = cv2.moments(bot_c)
                    if tM["m00"] > 0 and bM["m00"] > 0:
                        centroids = [
                            (int(tM["m10"]/tM["m00"]), int(tM["m01"]/tM["m00"]) + y),
                            (int(bM["m10"]/bM["m00"]), int(bM["m01"]/bM["m00"]) + mid_y),
                        ]
                        centroids.sort(key=lambda p: p[1])
                        return centroids
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

    # Reject if centroids are too close vertically (noise or split artifacts)
    dy = abs(centroids[0][1] - centroids[1][1])
    if dy < 18:
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


def _flush_camera(cap, count=CAMERA_FLUSH_FRAMES):
    """Grab and discard frames to flush the camera buffer.

    USB cameras buffer old frames internally. After the motor stops,
    the buffered frames still show the old position. Flushing ensures
    the next read() returns a current image.
    """
    for _ in range(count):
        cap.grab()


def _try_detect_markers(cap, roi_x_fraction, retries=3, retry_delay=0.2):
    """Try to detect markers, retrying on failure.

    Cameras sometimes return under-exposed or blurry frames right after
    the motor stops. Retrying a few times greatly improves reliability.

    Returns:
        (success: bool, frame, markers) — frame and markers are from the
        successful read, or from the last failed attempt.
    """
    frame = None
    for i in range(retries):
        success, frame = cap.read()
        if not success:
            return False, None, None
        markers = detect_markers(frame, roi_x_fraction=roi_x_fraction)
        if markers is not None:
            return True, frame, markers
        if i < retries - 1:
            time.sleep(retry_delay)
    return True, frame, None


def home(cap, nudge_speed=0.7, nudge_duration=0.3, tolerance_px=6,
         max_attempts=40, settle_time=0.5, move_fn=None,
         roi_x_fraction=ROI_X_FRACTION, flush_frames=CAMERA_FLUSH_FRAMES,
         confirm_reads=3, min_brightness=MIN_BRIGHTNESS):
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
        flush_frames: Number of frames to flush before first read.
        confirm_reads: Consecutive in-tolerance reads required to confirm alignment.

    Returns:
        Dict with keys: aligned (bool), final_error_px (int), attempts (int),
            marker_loss_count (int).
    """
    if move_fn is None:
        move_fn = motor.move

    # Flush stale buffered frames before starting
    _flush_camera(cap, flush_frames)

    # Check ambient brightness — low light shifts marker hue, skip homing
    if min_brightness > 0:
        success, frame = cap.read()
        if success and frame is not None:
            brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
            if brightness < min_brightness:
                print(f"Homing: skipped, too dark (brightness={brightness:.0f}, min={min_brightness})")
                return {'aligned': False, 'final_error_px': 0,
                        'attempts': 0, 'marker_loss_count': 0,
                        'skipped': 'low_light', 'brightness': brightness}

    error = 0
    marker_loss_count = 0
    consecutive_ok = 0
    for attempt in range(max_attempts):
        cam_ok, frame, markers = _try_detect_markers(cap, roi_x_fraction)
        if not cam_ok:
            print("Homing: camera read failed, aborting")
            return {'aligned': False, 'final_error_px': 0,
                    'attempts': attempt + 1, 'marker_loss_count': marker_loss_count}

        if markers is None:
            marker_loss_count += 1
            consecutive_ok = 0
            print(f"Homing: markers not found after retries (attempt {attempt + 1}/{max_attempts})")
            time.sleep(settle_time)
            continue

        raw_error = compute_alignment_error(markers)
        error = raw_error - ALIGNED_OFFSET
        print(f"Homing: error={error}px (raw={raw_error}px), markers={markers} (attempt {attempt + 1}/{max_attempts})")

        if abs(error) <= tolerance_px:
            consecutive_ok += 1
            if consecutive_ok >= confirm_reads:
                print(f"Homing: aligned (error={error}px, confirmed {confirm_reads}x, attempt {attempt + 1})")
                return {'aligned': True, 'final_error_px': error,
                        'attempts': attempt + 1, 'marker_loss_count': marker_loss_count}
            continue
        else:
            consecutive_ok = 0

        # Nudge motor to correct: scale nudge strength by error magnitude.
        direction = -1.0 if error > 0 else 1.0
        if abs(error) > 100:
            speed = nudge_speed
            duration = nudge_duration * 1.5
        elif abs(error) > 60:
            speed = nudge_speed
            duration = nudge_duration
        elif abs(error) > 30:
            speed = nudge_speed * 0.7
            duration = nudge_duration * 0.7
        else:
            speed = nudge_speed * 0.7
            duration = nudge_duration * 0.7
        move_fn(velocity=direction * speed, duration=duration)
        time.sleep(settle_time)

        # Flush frames after nudge so next read is current
        _flush_camera(cap, flush_frames)

    print(f"Homing: max attempts ({max_attempts}) reached, final error={error}px")
    return {'aligned': False, 'final_error_px': error,
            'attempts': max_attempts, 'marker_loss_count': marker_loss_count}
