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
Plot frequency distribution (and timestamp differences) per single Myo.
"""

import multiprocessing
import itertools

import pandas as pd
import numpy as np

from preprocess import utils

def get_timestamp_diff_and_freq(prefix, sensor):
	left = utils.read_csv(f"{prefix}.left.{sensor}.csv")
	left_f = utils.get_frequency(left["time"].values)
	left = left["time"].diff()
	right = utils.read_csv(f"{prefix}.right.{sensor}.csv")
	right_f = utils.get_frequency(right["time"].values)
	right = right["time"].diff()
	left = left[~left.isna()][(left < 1) & (left > 0)]
	right = right[~right.isna()][(right < 1) & (right > 0)]
	if sensor == "emg" and left_f < 199 or left_f > 200.2:
		print(prefix, "left", left_f)
	if sensor == "emg" and right_f < 199 or right_f > 200.2:
		print(prefix, "right", right_f)
	return left, right, left_f, right_f

def main(path: "path to a directory to load recordings from" = "train-data/", sensor="imu"):
	def predicate(prefix):
		meta = utils.read_meta(prefix)
		sync_errors = any("SYNC_ERROR_C" in note for note in meta["notes"])
		return not sync_errors
	args = itertools.product(utils.get_task_prefices(path, predicate), [sensor])

	with multiprocessing.Pool() as pool:
		left_diffs, right_diffs, left_freqs, right_freqs = zip(*pool.starmap(get_timestamp_diff_and_freq, args))

	print(f"left mean frequency: {np.mean(left_freqs)} (std: {np.std(left_freqs)}")
	print(f"right mean frequency: {np.mean(right_freqs)} (std: {np.std(right_freqs)}")
	print(f"mean frequency ratio: {np.mean(left_freqs) / np.mean(right_freqs)}")

	#new_left_diffs = pd.Series([])
	#for left_diff in left_diffs:
	#	new_left_diffs = new_left_diffs.append(left_diff, ignore_index=True)
	#print(len(new_left_diffs), new_left_diffs.mean(), 1 / new_left_diffs.mean())

	#new_right_diffs = pd.Series([])
	#for right_diff in right_diffs:
	#	new_right_diffs = new_right_diffs.append(right_diff, ignore_index=True)
	#print(len(new_right_diffs), new_right_diffs.mean(), 1 / new_right_diffs.mean())
