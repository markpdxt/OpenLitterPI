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
Command-line interface for manual motor control.
"""

import argparse
import motor


def main():
    parser = argparse.ArgumentParser(description="OpenLitterPI Motor Control")

    parser.add_argument(
        '--velocity', '-v',
        help='Motor speed and direction (-1.0 to 1.0)',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--duration', '-d',
        help='Duration in seconds',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--type', '-t',
        help='Operation type: "rotate" or "cycle"',
        choices=['rotate', 'cycle'],
        default='rotate'
    )

    args = parser.parse_args()

    if args.type == "rotate":
        motor.move(args.velocity, args.duration)
    elif args.type == "cycle":
        motor.cycle()


if __name__ == '__main__':
    main()
