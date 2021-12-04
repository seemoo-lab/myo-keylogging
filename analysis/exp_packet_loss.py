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

"""
Print all lost packets (based on EMG data) and all occurences of a premature sensor disconnect.
"""

import itertools
import multiprocessing

from preprocess import utils, fuse

def identify_interrupt(prefix, threshold):
	imu_left = utils.read_csv(f"{prefix}.left.imu.csv")
	imu_right = utils.read_csv(f"{prefix}.right.imu.csv")
	time_diff = fuse.get_time_diff(imu_left["time"].values, imu_right["time"].values)
	if time_diff > threshold:
		print(f"{prefix} - left and right duration differ by {time_diff:.2f} s")

def identify_packet_loss(prefix, arm):
	emg_data = utils.read_sensor(f"{prefix}.{arm}.emg.csv")
	if utils.check_packet_loss(emg_data["characteristics"]):
		with lock:
			print(f"{prefix}.{arm}.emg.csv")
			for i in range(2, len(emg_data["characteristics"]), 2):
				result = emg_data["characteristics"].iloc[i]
				expected = (emg_data["characteristics"].iloc[i - 2] + 1) % 4
				if result != expected:
					print(f"\tline: {i + 2} - expected {expected} got {result} (diff: {abs(expected - result)})")

def init(_lock):
	global lock
	lock = _lock

def main(
		path: "path to a directory to load recordings from" = "train-data/",
		threshold: "threshold for the maximum sensor duration difference" = 0.04,
	):
	args_packet_loss = itertools.product(sorted(utils.get_task_prefices(path)), ["left", "right"])
	args_interrupt = itertools.product(sorted(utils.get_task_prefices(path)), [threshold])
	lock = multiprocessing.Lock()
	with multiprocessing.Pool(initializer=init, initargs=(lock, )) as pool:
		print("Packet loss:")
		pool.starmap(identify_packet_loss, args_packet_loss)
		print("Sensor loss:")
		pool.starmap(identify_interrupt, args_interrupt)
