# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2019-2021  Matthias Gazzari, Annemarie Mattmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import types
import logging
import time
import functools
import argparse

import record
import myo_raw

LOG = logging.getLogger(__name__)

def wait_task(prefix, myo_names, shared, duration):
	try:
		start_time = time.time()
		while not shared.stop_flag.is_set() and time.time() < start_time + duration:
			time.sleep(0.1)
			print(f"run time: {time.time() - start_time:.2f} s", end="\r")
	except KeyboardInterrupt:
		print()

def main():
	parser = argparse.ArgumentParser(description="Sample recording script for testing.")
	parser.add_argument("prefix", help="the file prefix")
	parser.add_argument("--tag", metavar="STRING", help="the description of the recorded data")
	parser.add_argument("--pl_freq", type=float, default=50.0, help="the power line frequency in Hz (default=%(default)s)")
	parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (default=%(default)s)")
	parser.add_argument("--count", type=int, default=1, help="number of single measurements (default=%(default)s)")
	parser.add_argument("--duration", type=int, default=float("inf"), help="duration of a single measurement (default=%(default)s)")
	parser.add_argument("--device", action="append", metavar=("TTY", "MAC", "ID"), nargs=3, default=[],
			help="the Myo dongle TTY, the Myo MAC address and a custom identifier to distinguish Myos (e.g. first/second)")
	parser.add_argument("--notch_filter", action="store_true",
			help="enable the onboard 50 Hz and 60 Hz notch filter to suppress power line noise (default=%(default)s)")
	args = parser.parse_args()

	# define log level
	log_format = "[%(levelname)s/%(processName)s] %(message)s"
	log_level = max(2 - args.verbose, 0) * 10
	logging.basicConfig(level=log_level, format=log_format)

	# derive arguments and start recording data
	meta = types.SimpleNamespace(tag=args.tag, pl_freq=args.pl_freq)
	subscribe_args = {
		"emg_mode": myo_raw.EMGMode.RAW_FILTERED if args.notch_filter else myo_raw.EMGMode.RAW,
		"imu_mode": myo_raw.IMUMode.ON,
		"clf_state": myo_raw.CLFState.OFF,
		"battery": False,
	}
	task = functools.partial(wait_task, duration=args.duration)

	for i in range(args.count):
		prefix = record.enumerate_prefix(args.prefix)
		LOG.info(f"round {i + 1}/{args.count} started")
		record.run(prefix, "Escape", False, args.device, subscribe_args, meta, task)

if __name__ == "__main__":
	main()
