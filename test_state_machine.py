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
        reset_threshold=480.0,   # 8 min
        detected_timeout=300.0,  # 5 min
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

class TestOccupiedFramesDecrement:
    def test_frames_hold_in_detected_state(self, sm, clock):
        """Once DETECTED, occupied_frames should NOT decrement so
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

    def test_complete_resets_to_idle_after_timeout(self, sm, clock):
        self._reach_using(sm, clock)
        clock.advance(425)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE

        clock.advance(500)  # past reset_threshold (480s)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE


# --- DETECTED timeout and promotion ---

class TestDetectedTimeout:
    def test_detected_stays_during_use_threshold(self, sm, clock):
        """DETECTED should NOT reset after use_threshold (45s) anymore.
        It waits for detected_timeout (300s) to allow cats time in the box."""
        for _ in range(5):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Advance past use_threshold but before detected_timeout
        clock.advance(50)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.DETECTED  # still DETECTED, not IDLE

    def test_detected_promotes_to_waiting_after_timeout(self, sm, clock):
        """After detected_timeout (300s), DETECTED promotes to WAITING
        since the cat likely used the box but was only briefly visible."""
        for _ in range(5):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Advance past detected_timeout
        clock.advance(305)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING
        assert ("message", "WAITING") in actions

    def test_detected_promotion_leads_to_cycling(self, sm, clock):
        """After DETECTED promotes to WAITING, system should eventually
        reach CYCLING when wait_threshold is exceeded."""
        for _ in range(5):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Advance past wait_threshold (420s from last detection)
        clock.advance(425)
        actions = sm.process_frame(cat_detected=False)
        # First frame promotes DETECTED → WAITING
        assert sm.status == Status.WAITING

        # Next frame: now in WAITING, since_detected > wait_threshold → CYCLING
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("cycle", None) in actions


# --- Global reset ---

class TestGlobalReset:
    def test_any_state_resets_after_reset_threshold(self, sm, clock):
        for _ in range(16):
            sm.process_frame(cat_detected=True)
        assert sm.status == Status.USING

        clock.advance(500)  # past reset_threshold
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

        # 8+ minutes pass -> back to IDLE
        clock.advance(500)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.IDLE


# --- Double detection on cat exit ---

class TestDoubleDetection:
    def test_cat_exit_no_second_detected_notification(self, sm, clock):
        """Cat enters (DETECTED), goes inside box (invisible for 60s),
        then exits (visible again). Should NOT trigger a second DETECTED."""
        # Cat enters — first detection
        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert ("message", "DETECTED") in actions

        # Cat inside box, invisible for 60s
        clock.advance(60)
        sm.process_frame(cat_detected=False)
        # Should still be DETECTED (detected_timeout is 300s)
        assert sm.status == Status.DETECTED

        # Cat exits — visible again
        actions = sm.process_frame(cat_detected=True)
        # Should NOT get a second DETECTED message
        assert ("message", "DETECTED") not in actions
        assert sm.status == Status.DETECTED

    def test_cat_exit_after_long_stay_no_second_detected(self, sm, clock):
        """Cat enters, stays 4 minutes (invisible), exits. Still one DETECTED."""
        actions = sm.process_frame(cat_detected=True)
        assert ("message", "DETECTED") in actions

        # Cat inside box for 4 minutes
        clock.advance(240)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.DETECTED

        # Cat exits — visible again
        actions = sm.process_frame(cat_detected=True)
        assert ("message", "DETECTED") not in actions


class TestFullCycleEntryOnlyVisibility:
    def test_entry_detection_silence_exit_leads_to_cycle(self, sm, clock):
        """Simulate real scenario: cat seen on entry, invisible inside box,
        seen on exit, then system waits and cycles."""
        # Cat enters — brief detection (3 frames)
        sm.process_frame(cat_detected=True)
        sm.process_frame(cat_detected=True)
        sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Cat inside box — invisible for 3 minutes
        clock.advance(180)
        sm.process_frame(cat_detected=False)
        assert sm.status == Status.DETECTED

        # Cat exits — brief detection (2 frames)
        sm.process_frame(cat_detected=True)
        sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED

        # Cat gone — wait for detected_timeout (300s from last detection)
        clock.advance(305)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING
        assert ("message", "WAITING") in actions

        # Wait for wait_threshold (420s from last detection)
        clock.advance(120)  # total 425s from last detection
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("cycle", None) in actions

    def test_single_frame_entry_still_cycles(self, sm, clock):
        """Even a single detection frame should eventually lead to a cycle
        if no further activity is detected."""
        actions = sm.process_frame(cat_detected=True)
        assert sm.status == Status.DETECTED
        assert ("message", "DETECTED") in actions

        # No more detections — wait for detected_timeout
        clock.advance(305)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.WAITING

        # Wait for wait_threshold
        clock.advance(120)
        actions = sm.process_frame(cat_detected=False)
        assert sm.status == Status.COMPLETE
        assert ("cycle", None) in actions
