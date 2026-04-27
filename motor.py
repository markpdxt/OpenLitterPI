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
Motor control module for OpenLitterPI.
Handles DC motor operations via Adafruit Motor HAT.
"""

import time
import board
from adafruit_motorkit import MotorKit


def cycle(rotate1=58.0, rotate2=69.0, rotate3=8.0, speed=1.0, delay=2.0):
    """
    Execute a full cleaning cycle.
    
    Args:
        rotate1: Duration of forward rotation in seconds (sift)
        rotate2: Duration of reverse rotation in seconds (dump)
        rotate3: Duration of return rotation in seconds (home)
        speed: Motor speed (0.0 to 1.0)
        delay: Pause between phases in seconds
    """
    kit = MotorKit(i2c=board.I2C())

    # Phase 1: Forward rotation (sift litter)
    kit.motor3.throttle = speed
    time.sleep(rotate1)
    kit.motor3.throttle = 0.0
    time.sleep(delay)

    # Phase 2: Reverse rotation (dump waste)
    kit.motor3.throttle = -1 * speed
    time.sleep(rotate2)
    kit.motor3.throttle = 0.0
    time.sleep(delay)

    # Phase 3: Return to home position
    kit.motor3.throttle = speed
    time.sleep(rotate3)
    kit.motor3.throttle = 0.0


def move(velocity=1.0, duration=0.5):
    """
    Move motor for a specified duration.
    
    Args:
        velocity: Motor speed and direction (-1.0 to 1.0)
        duration: How long to run in seconds
    """
    kit = MotorKit(i2c=board.I2C())

    kit.motor3.throttle = float(velocity)
    time.sleep(float(abs(duration)))
    kit.motor3.throttle = 0.0
