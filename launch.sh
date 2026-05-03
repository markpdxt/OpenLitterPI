#!/bin/bash
#
# OpenLitterPI Launch Script
# ==========================
#
# INSTALLATION:
#   1. Place this file in your home directory: ~/launch.sh
#   2. Make it executable: chmod +x ~/launch.sh
#   3. Create ~/.env with your email credentials (see .env.example)
#
# AUTO-START ON BOOT:
#   Add the following line to your crontab (crontab -e):
#   @reboot sleep 120; ~/launch.sh > ~/launch.log 2>&1
#
#   The 120-second delay allows the camera and I2C bus to initialize.
#
# FOLDER STRUCTURE:
#   ~/
#   ├── launch.sh                    <-- This file
#   ├── launch.log                   <-- Auto-generated log output
#   ├── .env                         <-- Email credentials (not checked in)
#   └── projects/
#       └── openlitterpi/
#           ├── detect.py            <-- Main detection script
#           ├── utils.py             <-- Motor control & notifications
#           ├── move.py              <-- Manual motor control CLI
#           ├── cycle.py             <-- Standalone cycle test
#           ├── models/              <-- TFLite model files
#           ├── photos/              <-- State transition snapshots
#           └── openlitterpi-env/    <-- Python virtual environment
#

# Load email credentials
if [ -f ~/.env ]; then
    source ~/.env
fi

cd /home/pi/openlitterpi
source openlitterpi-env/bin/activate
python detect.py
