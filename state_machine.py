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
State machine for the OpenLitterPI detection system.

Manages the lifecycle: IDLE -> DETECTED -> USING -> WAITING -> CYCLING -> COMPLETE

Extracted from detect.py so it can be tested without camera/motor hardware.
"""

import time
from enum import Enum
from typing import List, Optional, Tuple


class Status(Enum):
    IDLE = 0
    DETECTED = 1
    USING = 2
    WAITING = 3
    CYCLING = 4
    COMPLETE = 5


class LitterBoxStateMachine:
    """
    Manages the litterbox detection state machine.

    Feed it detection results each frame via process_frame().
    It returns state transitions and action requests (send_message, cycle).
    """

    def __init__(
        self,
        occupied_frames_threshold: int = 15,
        use_threshold: float = 45.0,
        wait_threshold: float = 60.0 * 7,
        reset_threshold: float = 60.0 * 8,
        detected_timeout: float = 45.0,
        time_fn=None,
    ):
        self.occupied_frames_threshold = occupied_frames_threshold
        self.use_threshold = use_threshold
        self.wait_threshold = wait_threshold
        self.reset_threshold = reset_threshold
        self.detected_timeout = detected_timeout
        self._time_fn = time_fn or time.time

        self.status = Status.IDLE
        self.elapsed_time = 0
        self.occupied_frames = 0
        self._timestamp_last_detected: Optional[float] = None

    def _now(self) -> float:
        return self._time_fn()

    def _seconds_since_detected(self) -> float:
        if self._timestamp_last_detected is None:
            return float('inf')
        return self._now() - self._timestamp_last_detected

    def process_frame(self, cat_detected: bool) -> List[Tuple[str, Optional[str]]]:
        """
        Process a single frame's detection result.

        Args:
            cat_detected: True if at least one cat (or teddy bear) was
                          detected in this frame. Caller should collapse
                          multiple detections into a single boolean.

        Returns:
            List of (action, status_name) tuples:
              ("message", status_name) - send a notification
              ("cycle", None)          - run the motor cycle
        """
        actions = []

        if cat_detected:
            self.occupied_frames += 1
            self._timestamp_last_detected = self._now()

            if self.occupied_frames <= self.occupied_frames_threshold:
                # Only transition to DETECTED from IDLE or COMPLETE.
                # Don't regress from USING+ (e.g., after timeout promotion).
                if self.status in (Status.IDLE, Status.COMPLETE):
                    self.status = Status.DETECTED
                    actions.append(("message", self.status.name))
                self.elapsed_time = self.occupied_frames
            else:
                if self.status != Status.USING:
                    self.status = Status.USING
                    actions.append(("message", self.status.name))
                self.elapsed_time = self.occupied_frames

        # Time-based transitions (only when actively tracking)
        since_detected = self._seconds_since_detected()

        # DETECTED → USING promotion: cat was detected but became invisible
        # (likely entered the box). Promote to USING and reset the timer so
        # the use_threshold countdown starts fresh from this point.
        if self.status == Status.DETECTED and since_detected > self.detected_timeout:
            self.status = Status.USING
            self._timestamp_last_detected = self._now()
            actions.append(("message", self.status.name))
            self.elapsed_time = self.occupied_frames
            since_detected = self._seconds_since_detected()

        if self.status.value >= Status.USING.value:
            if since_detected > self.use_threshold and since_detected <= self.wait_threshold:
                if self.status != Status.WAITING:
                    self.status = Status.WAITING
                    actions.append(("message", self.status.name))
                self.elapsed_time = int(since_detected)
            elif since_detected > self.wait_threshold:
                if self.status != Status.CYCLING and self.status != Status.COMPLETE:
                    self.status = Status.CYCLING
                    actions.append(("message", self.status.name))
                    actions.append(("cycle", None))
                    self.status = Status.COMPLETE
                    actions.append(("message", self.status.name))
                    self.elapsed_time = 0
                    self.occupied_frames = 0

        # Global safety reset: if no detection for reset_threshold,
        # reset to IDLE to prevent being stuck in any state forever.
        if self.status != Status.IDLE and since_detected > self.reset_threshold:
            self._reset()

        return actions

    def _reset(self):
        self.status = Status.IDLE
        self.elapsed_time = 0
        self.occupied_frames = 0
        self._timestamp_last_detected = None
