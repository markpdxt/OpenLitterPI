# Plan: Visual Homing to Fix Bin Drift

## Context

The litterbox cleaning cycle uses timed motor phases: forward 54s (sift), reverse 65s (dump), forward 7.28s (return home). With no position sensor, the timed "return home" phase accumulates error over repeated cycles, causing the bin to drift off-level. The camera can see two physical markers — one on the bin, one on the frame — that line up when the bin is level.

## Problem Analysis

**Root cause**: `motor.py:27` — `cycle()` Phase 3 runs forward for exactly 7.28s. This open-loop timing has no feedback. Motor speed variations, voltage fluctuations, and mechanical resistance cause cumulative drift after repeated cycles.

**What we have**: USB webcam (640x480) with OpenCV already installed, two alignment markers visible to the camera, and `motor.move(velocity, duration)` for fine-grained nudges.

## Solution: Two-Marker Alignment Homing

After the timed cycle completes, capture frames, detect both markers, and nudge the motor until they are horizontally aligned. **No calibration file needed** — alignment is determined by the relative position of the two markers in each frame.

### How It Works

```
After motor.cycle() completes:
  1. Capture frame from camera
  2. HSV threshold -> find two colored marker contours -> get centroids
  3. Compare x-coordinates of the two centroids
  4. If |x_diff| < tolerance (~10px): DONE (markers aligned, bin is level)
  5. Else: motor.move(direction, 0.1s at 0.4 speed) to nudge
  6. Wait 0.3s for motor to settle
  7. Repeat (max 20 attempts)
  8. If max attempts exceeded: log warning, accept position (same as today)
```

Only the **x-axis difference** matters — the bin rotates around a single axis, so horizontal displacement between the two markers indicates rotational error. One marker is fixed (frame), one moves with the bin.

### Why Two-Marker Alignment

- No calibration step or config file needed — the frame marker IS the reference
- Self-correcting if camera shifts slightly over time
- Detection is simple: find two blobs of the same color, compare x-coordinates
- Distinguishing which marker is which: the frame marker is always stationary, the bin marker moves. We can identify them by vertical position (one is always above the other) or just use the x-difference regardless of which is which.

## Implementation

### New Files

**`homing.py`** — core visual homing module (~80 lines)
- `detect_markers(frame, hsv_lower, hsv_upper, min_area=100)` — HSV threshold, find two largest contours of the marker color, return their centroids sorted by y-coordinate (top, bottom), or `None` if <2 found
- `compute_alignment_error(markers)` — returns x-difference between the two marker centroids
- `home(cap, nudge_speed=0.4, nudge_duration=0.1, tolerance_px=10, max_attempts=20, settle_time=0.3)` — closed-loop homing routine. Returns `True` if aligned, `False` if max attempts exceeded.
- HSV range for the marker color stored as module constants (tuned during physical setup)

**`test_homing.py`** — unit tests (~100 lines)
- Test `detect_markers()` with synthetic images (draw two colored circles at known positions)
- Test `compute_alignment_error()` with various centroid pairs
- Test `home()` loop logic with mock camera (returns synthetic frames with shifting marker positions) and mock motor

### Modified Files

**`detect.py:100-104`** — add homing after cycle (2 lines)
```python
# Current:
elif action == "cycle":
    utils.cycle()

# New:
elif action == "cycle":
    utils.cycle()
    homing.home(cap)
```

**`test_hardware.py:138-139`** — add `homing.home(cap)` after motor cycle in integration test

**`CLAUDE.md`** — document homing system, marker placement, and HSV tuning

### Unchanged Files
- `motor.py` — `move()` already provides fine-grained control needed for nudges
- `state_machine.py` — pure logic, unaffected
- `utils.py` — no changes needed
- `requirements.txt` — no new dependencies

## Sequence

```
state_machine emits ("cycle", None)
    |
    v
motor.cycle()                     # Phases 1-3, blocking ~130s
    |                             # Camera idle but handle (cap) stays open
    v
homing.home(cap)                  # Visual alignment, ~2-10s typically
    |-- cap.read() -> frame
    |-- detect_markers(frame) -> [(x1,y1), (x2,y2)]
    |-- x_diff = x1 - x2
    |-- if |x_diff| < 10px: DONE
    |-- else: motor.move(+/-0.4, 0.1s)
    |-- sleep(0.3s)
    |-- repeat up to 20x
    |
    v
Back to main detection loop      # Cat detection resumes
```

## Failure Modes

| Scenario | Behavior | Impact |
|----------|----------|--------|
| Marker(s) not visible | `home()` returns False, logs warning | Same as today (timed Phase 3 position) |
| Camera failure | `cap.read()` returns False, abort | System continues normally |
| Motor overshoot | Small nudges (0.1s/40% speed) self-correct next iteration | Adds 1-2 extra iterations |
| Lighting changes | Fluorescent tape has strong saturation in HSV | Robust to typical indoor variation |
| Only 1 marker found | Treated same as "not visible" — need both | Falls back to timed position |

## Physical Setup

1. Attach two pieces of the same bright-colored tape (fluorescent green or orange recommended) — one on the bin, one on the stationary frame — positioned so they align horizontally when the bin is level
2. Ensure the tape color doesn't match anything else in camera view
3. Tune HSV constants in `homing.py` if needed (we'll provide guidance for this)

## Verification

1. `python3 -m pytest test_state_machine.py -v` — existing tests still pass (no state machine changes)
2. `python3 -m pytest test_homing.py -v` — new unit tests with synthetic images (no hardware needed)
3. On Pi: manually test marker detection by running a quick script that captures a frame and prints detected centroids
4. On Pi: `python3 test_hardware.py` — full cycle with homing
5. Monitor over 5+ cycles to confirm drift is eliminated
