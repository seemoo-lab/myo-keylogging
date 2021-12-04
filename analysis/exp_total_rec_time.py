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
Print the total recording time.
"""

import multiprocessing

from preprocess import utils

def min_imu_time(prefix):
	left_imu_data = utils.read_sensor(f"{prefix}.left.imu.csv")
	right_imu_data = utils.read_sensor(f"{prefix}.right.imu.csv")
	left_time = left_imu_data.index.values[-1] - left_imu_data.index.values[0]
	right_time = right_imu_data.index.values[-1] - right_imu_data.index.values[0]
	return min(left_time, right_time)

def key_time(prefix):
	key_data = utils.read_csv(f"{prefix}.key.csv")
	return key_data["time"].iloc[-1] - key_data["time"].iloc[0]

def key_strokes(prefix):
	key_data = utils.read_csv(f"{prefix}.key.csv")
	return key_data["event"].value_counts()["release"]

def main(path: "path to a directory to load recordings from" = "train-data/"):
	args = utils.get_task_prefices(path)
	with multiprocessing.Pool() as pool:
		tot_key_time = sum(pool.map(key_time, args))
		tot_imu_time = sum(pool.map(min_imu_time, args))
		tot_key_num = sum(pool.map(key_strokes, args))
	print("total recording time:")
	print(f"key data based: {tot_key_time:.0f} s ({tot_key_time / 3600:.2f} h)")
	print(f"imu data based: {tot_imu_time:.0f} s ({tot_imu_time / 3600:.2f} h)")
	print(f"total number of keystrokes: {tot_key_num}")
