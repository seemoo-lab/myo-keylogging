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
Print file duration, index and file name of timestamp differences larger than a given threshold.
"""

import multiprocessing
import json
import pprint
import itertools

from preprocess import utils

def print_stats(file_name, max_diff, idx, sync_error_c):
	pprint.pprint({
		"file name": file_name,
		"time diff":  f"{max_diff:.3f} s",
		"line num": idx,
		"sync_error_c": sync_error_c,
	}, indent=4)

def identify_outlier(prefix, threshold, arm, filter_sync_error_c):
	meta = json.load(open(f"{prefix}.meta.json"))
	sync_error_c = any("SYNC_ERROR_C" in note for note in meta["notes"])
	file_name = f"{prefix}.{arm}.imu.csv"
	if filter_sync_error_c and sync_error_c:
		return 0, 0, file_name, sync_error_c

	imu_time_diff = utils.read_csv(file_name)["time"].diff()
	idx = imu_time_diff.idxmax()
	max_diff = imu_time_diff.iloc[idx]
	return max_diff, idx, file_name, sync_error_c

def main(
		path: "path to a directory to load recordings from" = "train-data/",
		threshold: "threshold for outlier" = 0.02 * 4,
		filter_sync_error_c: "filter SYNC_ERROR_C" = True
	):
	prefices = sorted(utils.get_task_prefices(path))
	args = itertools.product(prefices, [threshold], ["left", "right"], [filter_sync_error_c])
	with multiprocessing.Pool() as pool:
		result = pool.starmap(identify_outlier, args)

	for max_diff, idx, file_name, sync_error_c in result:
		if max_diff > threshold:
			print_stats(file_name, max_diff, idx, sync_error_c)
