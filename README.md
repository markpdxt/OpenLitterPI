# OpenLitterPI

An open-source, automated cat litterbox powered by Raspberry PI and TensorFlow Lite. Uses computer vision to detect when your cat uses the litterbox, waits an appropriate time after they leave, then automatically cycles a self-cleaning mechanism. Tested with a Litter-Robot 3

## Features

- **Real-time cat detection** using TensorFlow Lite and EfficientDet Lite0
- **Smart timing** — waits 7 minutes after cat leaves before cleaning
- **Email notifications** with photos at each state change
- **Auto-start on boot** — runs headlessly without intervention
- **DC motor control** via Adafruit Motor HAT

## How It Works

OpenLitterPI uses a state machine to manage the cleaning cycle:

```
IDLE → DETECTED → USING → WAITING → CYCLING → COMPLETE
```

1. **IDLE**: Monitoring for cat presence
2. **DETECTED**: Cat detected (< 15 frames)
3. **USING**: Cat confirmed using litterbox (15+ frames)
4. **WAITING**: Cat left, waiting 7 minutes before cleaning
5. **CYCLING**: Motor rotating to sift and clean
6. **COMPLETE**: Cycle finished, returns to IDLE

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Raspberry PI** | Raspberry PI 4 Model B (2GB+ RAM recommended) |
| **Motor HAT** | Adafruit DC & Stepper Motor HAT (PCA9685) |
| **Camera** | USB webcam (640×480 minimum) |
| **Motor** | DC gearmotor for drum rotation |
| **Power** | Adequate power supply for PI + motor |

### Wiring

- Motor connected to **Motor 3 (M3)** port on the Adafruit HAT
- Camera connected via USB
- Motor HAT communicates via I2C (address `0x60`)

## Software Requirements

- Raspbian GNU/Linux 11 (Bullseye) or later
- Python 3.9+
- I2C enabled on Raspberry PI

## Installation

### 1. Clone the repository

```bash
cd ~/projects
git clone https://github.com/yourusername/openlitterpi.git
cd openlitterpi
```

### 2. Create virtual environment

```bash
python3 -m venv openlitterpi-env
source openlitterpi-env/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Enable I2C

```bash
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable
```

### 5. Configure email notifications

Copy the example environment file and edit with your credentials:

```bash
cp .env.example .env
nano .env
```

Set these environment variables:
```
OPENLITTERPI_EMAIL_FROM=your-email@gmail.com
OPENLITTERPI_EMAIL_PASSWORD=your-app-password
OPENLITTERPI_EMAIL_TO=recipient@example.com
```

Then source the file before running:
```bash
source .env
```

> **Note**: Use a Gmail App Password, not your regular password. Generate one at [Google Account → Security → App Passwords](https://myaccount.google.com/apppasswords).

### 6. Set up auto-start (optional)

Place `launch.sh` in your home directory and add to crontab:

```bash
cp launch.sh ~/launch.sh
chmod +x ~/launch.sh
crontab -e
```

Add this line:

```
@reboot sleep 120; ~/launch.sh > ~/launch.log 2>&1
```

## Usage

### Manual start

```bash
cd ~/projects/openlitterpi
source openlitterpi-env/bin/activate
python3 detect.py
```

### With display (for debugging)

```bash
python3 detect.py --display True
```

### Manual motor control

```bash
# Rotate forward for 5 seconds
python3 move.py -v 1.0 -d 5.0

# Rotate backward for 5 seconds  
python3 move.py -v -1.0 -d 5.0

# Run full cleaning cycle
python3 move.py -t cycle
```

### Test cycle only

```bash
python3 cycle.py
```

## Configuration

### Timing parameters (in `detect.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `occupied_frames_threshold` | 15 | Frames to confirm cat presence |
| `use_threshold` | 45 sec | Time after cat leaves before WAITING |
| `wait_threshold` | 7 min | Time to wait before cycling |
| `reset_threshold` | 8 min | Timeout to reset to IDLE |

### Motor cycle parameters (in `utils.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rotate1` | 54 sec | Forward rotation (sift) |
| `rotate2` | 62 sec | Reverse rotation (dump) |
| `rotate3` | 7.28 sec | Return to home position |
| `speed` | 1.0 | Motor speed (0.0 - 1.0) |

## Project Structure

```
openlitterpi/
├── detect.py           # Main detection & state machine
├── utils.py            # Motor control, email, visualization
├── move.py             # Manual motor control CLI
├── cycle.py            # Standalone cycle test
├── launch.sh           # Auto-start script
├── models/
│   ├── efficientdet_lite0.tflite        # Standard TFLite model
│   └── efficientdet_lite0_edgetpu.tflite # EdgeTPU model (optional)
└── photos/             # State transition snapshots (auto-generated)
```

## Troubleshooting

### Camera not detected

```bash
# List video devices
v4l2-ctl --list-devices

# Check if camera is recognized
lsusb | grep -i cam
```

### I2C not working

```bash
# Check if I2C is enabled
sudo i2cdetect -y 1

# Should show device at address 0x60
```

### Motor not responding

- Verify motor is connected to M3 port
- Check power supply is adequate
- Test with `python3 cycle.py`

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full license text.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [TensorFlow Lite](https://www.tensorflow.org/lite) for the object detection framework
- [Adafruit](https://www.adafruit.com/) for the excellent Motor HAT and CircuitPython libraries
- The Raspberry PI Foundation
