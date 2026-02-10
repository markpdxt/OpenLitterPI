# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the LitterBoxStateMachine."""

import pytest
from state_machine import LitterBoxStateMachine, Status


class FakeClock:
    """Controllable clock for deterministic testing."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def sm(clock):
    return LitterBoxStateMachine(
        occupied_frames_threshold=15,
        use_threshold=45.0,
        wait_threshold=420.0,    # 7 min
        reset_threshold=480.0,   # 8 min global safety reset
        detected_timeout=45.0,   # 45s DETECTED → USING promotion
        time_fn=clock,
    )


# --- Basic state transitions ---

class TestIdleToDetected:
    def test_first_cat_detection_transitions_to_detected(self, sm, clock):
        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert ("message", "DETECTED") in actions

    def test_no_detection_stays_idle(self, sm, clock):
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE
        assert actions == []

    def test_repeated_detection_stays_detected_until_threshold(self, sm, clock):
        sm.process_frame(cat_detected=True)  # frame 1
        for i in range(2, 16):  # frames 2-15
            actions = sm.process_frame(cat_detected=True)
            assert sm.status == Status.DETECTED
            # No duplicate DETECTED messages
            assert ("message", "DETECTED") not in actions


class TestDetectedToUsing:
    def test_crosses_threshold_to_using(self, sm, clock):
        for _ in range(15):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        actions = sm.process_frame(cat_detected=True)  # frame 16
        assert sm.status == Status.USING
        assert ("message", "USING") in actions

    def test_occupied_frames_tracks_count(self, sm, clock):
        for _ in range(10):
            sm.process_frame(cat_detected=True)
        assert sm.occupied_frames == 10
        assert sm.elapsed_time == 10


# --- Frame counter behavior ---

class TestOccupiedFramesAccumulation:
    def test_frames_hold_in_detected_state(self, sm, clock):
        """Once DETECTED, occupied_frames should hold steady so
        intermittent camera detections can accumulate toward USING."""
        for _ in range(10):
            sm.process_frame(cat_detected=True)
        assert sm.occupied_frames == 10
        assert sm.status == Status.DETECTED

        for _ in range(5):
            sm.process_frame(cat_detected=False)
        # Frames hold steady — no decrement in DETECTED state
        assert sm.occupied_frames == 10

    def test_intermittent_detection_accumulates_in_detected(self, sm, clock):
        """Cat seen for 5 frames, gone for 10, seen for 5 more.
        Frames should accumulate since we're in DETECTED state."""
        for _ in range(5):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert sm.occupied_frames == 5

        for _ in range(10):
            sm.process_frame(cat_detected=False)
        assert sm.occupied_frames == 5  # held, not decremented

        for _ in range(5):
            sm.process_frame(cat_detected=True)
        assert sm.occupied_frames == 10

    def test_intermittent_detection_reaches_using(self, sm, clock):
        """Simulates a real camera: cat present but detected ~60% of frames.
        Should still reach USING state."""
        import random
        random.seed(42)
        for _ in range(50):
            detected = random.random() < 0.6
            sm.process_frame(cat_detected=detected)
        # With 60% detection over 50 frames, ~30 detections should
        # easily pass the 15-frame threshold
        assert sm.status == Status.USING


# --- Multi-detect per frame ---

class TestMultiDetectCollapse:
    def test_caller_collapses_multiple_detections(self, sm, clock):
        """Even if there are 3 cat bounding boxes, process_frame
        is called once with cat_detected=True. Counter increments by 1."""
        sm.process_frame(cat_detected=True)
        assert sm.occupied_frames == 1


# --- USING -> WAITING -> CYCLING -> COMPLETE flow ---

class TestFullCycle:
    def _reach_using(self, sm, clock):
        for _ in range(16):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.USING

    def test_using_to_waiting(self, sm, clock):
        self._reach_using(sm, clock)
        # Cat leaves, advance past use_threshold (45s)
        clock.advance(50)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING
        assert ("message", "WAITING") in actions

    def test_waiting_to_cycling_and_complete(self, sm, clock):
        self._reach_using(sm, clock)
        clock.advance(425)  # past wait_threshold (420s)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("message", "CYCLING") in actions
        assert ("cycle", None) in actions
        assert ("message", "COMPLETE") in actions
        assert sm.occupied_frames == 0
        assert sm.elapsed_time == 0

    def test_complete_resets_to_idle_after_reset_threshold(self, sm, clock):
        """After COMPLETE, global reset returns to IDLE."""
        self._reach_using(sm, clock)
        clock.advance(425)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE

        clock.advance(500)  # past reset_threshold (480s from last detection)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE

    def test_new_detection_after_complete_goes_to_detected(self, sm, clock):
        """A new cat after COMPLETE should start fresh at DETECTED."""
        self._reach_using(sm, clock)
        clock.advance(425)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert sm.occupied_frames == 0

        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert sm.occupied_frames == 1
        assert ("message", "DETECTED") in actions


# --- Happy path end-to-end ---

class TestHappyPath:
    def test_full_detection_to_cycle_to_idle(self, sm, clock):
        """Simulate: cat enters, uses box, leaves, system waits, cycles, resets."""
        # Cat enters and is detected
        sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Cat stays for 16 frames -> USING
        for _ in range(15):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.USING

        # Cat leaves, 50 seconds pass -> WAITING
        clock.advance(50)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING

        # 7+ minutes total pass -> CYCLING -> COMPLETE
        clock.advance(375)  # total ~425s from last detection
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("cycle", None) in actions

        # 8+ minutes pass -> global reset back to IDLE
        clock.advance(500)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE


# --- Global safety reset ---

class TestGlobalReset:
    def test_any_state_resets_after_reset_threshold(self, sm, clock):
        """Any non-IDLE state resets to IDLE after reset_threshold."""
        for _ in range(16):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.USING

        clock.advance(500)  # past reset_threshold (480s)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE
        assert sm.occupied_frames == 0
        assert sm.elapsed_time == 0

    def test_idle_does_not_reset(self, sm, clock):
        """IDLE state should not trigger reset logic."""
        clock.advance(1000)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE
        assert actions == []


# --- DETECTED → USING timeout promotion ---

class TestDetectedToUsingTimeout:
    def test_detected_stays_before_timeout(self, sm, clock):
        """DETECTED should persist before detected_timeout expires."""
        sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        clock.advance(30)  # less than 45s timeout
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.DETECTED

    def test_detected_promotes_to_using_after_timeout(self, sm, clock):
        """After detected_timeout (45s), DETECTED promotes to USING.
        Cat entered the box and became invisible to the camera."""
        sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        clock.advance(50)  # past 45s timeout
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.USING
        assert ("message", "USING") in actions

    def test_promotion_resets_timer(self, sm, clock):
        """After promotion, the use_threshold timer starts fresh.
        System should NOT immediately cascade to WAITING."""
        sm.process_frame(cat_detected=True)
        clock.advance(50)  # promote to USING
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.USING

        # 30s later — within use_threshold, should stay USING
        clock.advance(30)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.USING

    def test_promotion_eventually_cycles(self, sm, clock):
        """Full flow: DETECTED → USING (timeout) → WAITING → CYCLING."""
        sm.process_frame(cat_detected=True)
        clock.advance(50)  # promote to USING
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.USING

        clock.advance(50)  # past use_threshold from promotion
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING

        clock.advance(375)  # past wait_threshold from promotion
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("cycle", None) in actions


class TestDoubleDetection:
    def test_cat_exit_no_second_detected_notification(self, sm, clock):
        """Cat enters (DETECTED), goes inside box (invisible for 30s),
        then exits (visible again). Should NOT trigger a second DETECTED."""
        # Cat enters — first detection
        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert ("message", "DETECTED") in actions

        # Cat inside box, invisible for 30s (before detected_timeout)
        clock.advance(30)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.DETECTED

        # Cat exits — visible again
        actions = sm.process_frame(cat_detected=True)
        # Should NOT get a second DETECTED message
        assert ("message", "DETECTED") not in actions
        assert sm.status == Status.DETECTED

    def test_no_regression_to_detected_after_timeout_promotion(self, sm, clock):
        """After DETECTED → USING timeout promotion, a new detection
        should NOT regress back to DETECTED."""
        actions = sm.process_frame(cat_detected=True)
        assert ("message", "DETECTED") in actions

        # Cat inside box — promoted to USING after timeout
        clock.advance(50)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.USING

        # Cat exits — visible again (occupied_frames still < 15)
        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.USING  # stays USING, no regression
        assert ("message", "DETECTED") not in actions
