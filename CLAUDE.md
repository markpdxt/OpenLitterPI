# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenLitterPI is an automated cat litterbox using a Raspberry Pi, USB webcam, TensorFlow Lite (EfficientDet Lite0), and a DC motor via Adafruit Motor HAT. It detects cats via computer vision, waits after they leave, then cycles a self-cleaning mechanism.

## Commands

### Run tests (no hardware needed)
```bash
python3 -m pytest test_state_machine.py -v
```

### Run all tests
```bash
python3 -m pytest -v
```

### Run a single test
```bash
python3 -m pytest test_state_machine.py::TestOccupiedFramesDecrement::test_intermittent_detection_reaches_using -v
```

### Run detection (on Pi with camera)
```bash
python3 detect.py
```

### Run hardware integration test (on Pi)
```bash
python3 test_hardware.py
```

## Raspberry Pi Deployment

- **Host**: `pi@192.168.5.13`
- **SSH port**: `52222`
- **Project path**: `~/openlitterpi`
- **Repo is private** - use SCP to deploy files (git fetch won't work without auth on Pi)

```bash
# Deploy files
scp -P 52222 <files> pi@192.168.5.13:~/openlitterpi/

# Run tests on Pi
ssh -p 52222 pi@192.168.5.13 "cd ~/openlitterpi && python3 -m pytest test_state_machine.py -v"
```

## Architecture

### State Machine (core logic)

```
IDLE → DETECTED → USING → WAITING → CYCLING → COMPLETE
```

`state_machine.py` contains `LitterBoxStateMachine` - the pure-logic state machine extracted for testability. It has no hardware dependencies. Feed it `process_frame(cat_detected: bool)` each frame and it returns actions: `("message", status_name)` or `("cycle", None)`. After COMPLETE, status persists until a new cat detection restarts the cycle at DETECTED.

### Data Flow

```
USB Camera → detect.py (main loop) → TFLite model → cat_detected boolean
    → LitterBoxStateMachine.process_frame() → actions list
    → notifications.py (email + photo) and/or motor.py (cleaning cycle)
```

`detect.py` is the orchestrator: captures frames, runs inference, feeds the state machine, executes returned actions. `utils.py` re-exports `cycle()`/`move()`/`send_message()` and provides bounding-box visualization.

### Key Design Decisions

- **State machine is decoupled from hardware** - `state_machine.py` accepts an injectable `time_fn` for deterministic testing with `FakeClock`
- **Frame counting never decrements** - once a detection increments `occupied_frames`, it holds steady so intermittent camera detections accumulate toward the threshold (15 frames), matching original behavior
- **DETECTED promotes to USING after timeout** - if the cat is detected but then becomes invisible (entered the box), after `detected_timeout` (45s) the system promotes to USING and resets the timer so use_threshold starts fresh
- **Detection is boolean per frame** - multiple detections in a single frame are collapsed to a single `True` in `detect.py` before reaching the state machine
- **Motor on M3 port** - three-phase cycle: forward 54s (sift), reverse 65s (dump), forward 7.28s (home)

### Thresholds (production defaults in detect.py)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `occupied_frames_threshold` | 15 | Frames to confirm cat presence (DETECTED → USING) |
| `use_threshold` | 45s | No-detection time after USING before WAITING |
| `detected_timeout` | 45s | No-detection time in DETECTED before promoting to USING |
| `wait_threshold` | 420s (7 min) | No-detection time before CYCLING |
| `reset_threshold` | 480s (8 min) | Global safety timeout back to IDLE |

### Test Structure

- `test_state_machine.py` - comprehensive unit tests using `FakeClock` for time control, no hardware needed
- `test_hardware.py` - integration test on Pi with shortened thresholds (~30s full cycle)
- `tests/test_basic.py` - import and configuration validation
