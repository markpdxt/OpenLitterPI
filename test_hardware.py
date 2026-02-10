# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Hardware integration test for OpenLitterPI.

Run this ON THE PI with a teddy bear or cat in front of the camera.
Uses shortened thresholds so the full cycle completes in ~30 seconds.

Usage:
    cd /home/pi/openlitterpi
    source openlitterpi-env/bin/activate
    python test_hardware.py

    IMPORTANT: Stop the running detector first!
        kill $(pgrep -f 'python detect.py')

Flow:
    1. Camera starts, model loads
    2. Hold a teddy bear / cat in front of camera for ~5 seconds
    3. Remove the object
    4. Watch it progress: DETECTED -> USING -> WAITING -> CYCLING -> COMPLETE -> IDLE
    5. Motor will spin during CYCLING phase
    6. Script exits with PASS/FAIL summary
"""

import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
from state_machine import LitterBoxStateMachine, Status
import utils

TARGET_LABELS = {'cat', 'teddy bear'}

# Shortened thresholds for quick testing
TEST_CONFIG = {
    'occupied_frames_threshold': 3,   # 3 frames to confirm detection
    'use_threshold': 5.0,             # 5 sec after cat leaves -> WAITING
    'wait_threshold': 10.0,           # 10 sec after cat leaves -> CYCLING
    'reset_threshold': 15.0,          # 15 sec after cat leaves -> IDLE
}

TIMEOUT = 200  # max seconds (motor cycle takes ~127s)


def run_hardware_test(model='models/efficientdet_lite0.tflite', camera_id=0):
    sm = LitterBoxStateMachine(**TEST_CONFIG)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("FAIL: Cannot open camera")
        return False

    base_options = core.BaseOptions(
        file_name=model, use_coral=False, num_threads=4)
    detection_options = processor.DetectionOptions(
        max_results=5, score_threshold=0.35)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    print("=" * 50)
    print("  OpenLitterPI Hardware Integration Test")
    print("=" * 50)
    print()
    print("  Thresholds (shortened for testing):")
    print(f"    Frames to confirm:  {TEST_CONFIG['occupied_frames_threshold']}")
    print(f"    Use threshold:      {TEST_CONFIG['use_threshold']}s")
    print(f"    Wait threshold:     {TEST_CONFIG['wait_threshold']}s")
    print(f"    Reset threshold:    {TEST_CONFIG['reset_threshold']}s")
    print()
    print("  INSTRUCTIONS:")
    print("    1. Hold a teddy bear or cat in front of the camera")
    print("    2. Wait for USING status")
    print("    3. Remove the object")
    print("    4. Watch the full cycle complete")
    print()
    print("  Press Ctrl+C to abort at any time")
    print("=" * 50)
    print()

    states_reached = set()
    motor_cycled = False
    test_start = time.time()
    prev_status = None

    try:
        while cap.isOpened():
            elapsed = time.time() - test_start
            if elapsed > TIMEOUT:
                print(f"\nTIMEOUT after {TIMEOUT}s")
                break

            success, image = cap.read()
            if not success:
                print("FAIL: Cannot read from camera")
                break

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = vision.TensorImage.create_from_array(rgb_image)
            detection_result = detector.detect(input_tensor)

            cat_detected = any(
                d.categories[0].category_name in TARGET_LABELS
                for d in detection_result.detections
            )

            actions = sm.process_frame(cat_detected)

            # Log state changes
            if sm.status != prev_status:
                states_reached.add(sm.status)
                marker = ">>>" if sm.status != Status.IDLE else "---"
                print(f"  {marker} [{elapsed:5.1f}s] {sm.status.name}"
                      f"  (frames: {sm.occupied_frames})")
                prev_status = sm.status

            # Track transient states from actions (CYCLING transitions
            # to COMPLETE in the same process_frame call, so sm.status
            # is never observed as CYCLING)
            for action, status_name in actions:
                if action == "message" and status_name == "CYCLING":
                    states_reached.add(Status.CYCLING)
                    print(f"  >>> [{elapsed:5.1f}s] CYCLING  (frames: {sm.occupied_frames})")
                if action == "cycle":
                    print(f"  *** [{elapsed:5.1f}s] MOTOR CYCLING ***")
                    utils.cycle()
                    motor_cycled = True

            # Done once motor has cycled and we're back to IDLE
            if motor_cycled and sm.status == Status.IDLE:
                print(f"\n  --- [{elapsed:5.1f}s] Back to IDLE. Test complete.")
                break

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    finally:
        cap.release()

    # Report results
    print()
    print("=" * 50)
    print("  TEST RESULTS")
    print("=" * 50)

    expected_states = {Status.DETECTED, Status.USING, Status.WAITING,
                       Status.CYCLING, Status.COMPLETE}
    missing = expected_states - states_reached

    for state in [Status.IDLE, Status.DETECTED, Status.USING,
                  Status.WAITING, Status.CYCLING, Status.COMPLETE]:
        check = "PASS" if state in states_reached else "MISS"
        print(f"    [{check}] {state.name}")

    print(f"    [{'PASS' if motor_cycled else 'MISS'}] Motor cycle executed")
    print()

    if not missing and motor_cycled:
        print("  RESULT: PASS - Full hardware cycle verified")
        return True
    else:
        if missing:
            print(f"  RESULT: FAIL - Missing states: {', '.join(s.name for s in missing)}")
        if not motor_cycled:
            print("  RESULT: FAIL - Motor did not cycle")
        return False


if __name__ == '__main__':
    success = run_hardware_test()
    sys.exit(0 if success else 1)
