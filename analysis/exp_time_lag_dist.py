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
Plot the time lag distribution between two Myos (time diff and xcorr methods).
"""

import multiprocessing
import json

import numpy as np
import pandas as pd
from . import utils as plot_utils

from preprocess import utils, fuse

def estimate_time_lag(prefix):
	meta = json.load(open(f"{prefix}.meta.json"))
	if meta["sync"]:
		# read IMU data and set time column as index
		left_imu_data = utils.read_sensor(f"{prefix}.left.imu.csv")
		right_imu_data = utils.read_sensor(f"{prefix}.right.imu.csv")
		# determine measured lag
		diff_lag, sync_time = fuse.find_joint_acc_lag(
			left_imu_data,
			right_imu_data,
			meta["sync"]["min_acc_magnitude"],
			meta["sync"]["max_lag"]
		)
		# determine cross-correlation lag
		xcorr_lag, _ = fuse.xcorr_acc_norm(
			left_imu_data[left_imu_data.index < sync_time + 0.1],
			right_imu_data[right_imu_data.index < sync_time + 0.1]
		)
		return diff_lag, xcorr_lag / 50
	return np.nan, np.nan

def main(path: "path to a directory to load recordings from"="train-data/"):
	args = utils.get_task_prefices(path)
	with multiprocessing.Pool() as pool:
		diff_lag, xcorr_lag = zip(*pool.map(estimate_time_lag, args))
	diff_lag, xcorr_lag = np.array(diff_lag), np.array(xcorr_lag)

	# broken clap sync statistics
	count_diff_lag_nan = len(diff_lag[np.isnan(diff_lag)])
	count_xcorr_lag_nan = len(xcorr_lag[np.isnan(xcorr_lag)])
	print("num broken clap sync:", count_diff_lag_nan - count_xcorr_lag_nan)
	print("num without clap sync:", count_xcorr_lag_nan)

	# create data frame of non-nan values
	df = pd.DataFrame({
		"peak time difference (s)": diff_lag[~np.isnan(diff_lag)],
		"cross-correlation lag (s)": xcorr_lag[~np.isnan(xcorr_lag)]
	})

	# remove cross-correlation outlier and show how many have been removed
	df_without_outlier = df[
		(df["cross-correlation lag (s)"] >= df["peak time difference (s)"].min()) &
		(df["cross-correlation lag (s)"] <= df["peak time difference (s)"].max())
	]
	print("Remove cross-correlation outlier...")
	print("outlier:", len(df) - len(df_without_outlier))
	print(f"out of total: {len(df)} (relative:{(len(df) - len(df_without_outlier))/len(df):.4f})")

	# direct measure of the lag between high acceleration events
	plot_utils.plotf(plot_utils.plot_time_lag, data=df["peak time difference (s)"], path="results/analysis", filename="time_lag_dist")

