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

import logging
import pandas as pd
import numpy as np
from . import utils

LOG = logging.getLogger(__name__)

def xcorr_acc_norm(left, right, gravity=2048):
	"""
	Cross-correlate the absolute value of the accelerometer of two Myos and determine their lag.

	:param left: sensor data of the left Myo
	:param right: sensor data of the right Myo
	:param gravity: default acceleration magnitude of a stationary accelerometer
	:return: lag (number of samples) and the cross-correlation values
	"""
	col_names = ["acc0", "acc1", "acc2"]
	left_acc_norm = (left[col_names]**2).sum(1).pow(1/2)
	right_acc_norm = (right[col_names]**2).sum(1).pow(1/2)
	xcorr = np.correlate(left_acc_norm - gravity, right_acc_norm - gravity, 'full')
	lag = np.abs(xcorr).argmax() - len(xcorr) // 2
	return lag, xcorr

def find_joint_acc_lag(left, right, min_acc_mag, max_lag):
	"""
	Find time difference of the first occurence of a joint (two Myos) high acceleration event.

	:param left: left Myo sensor data
	:param right: right Myo sensor data
	:param min_acc_mag: minimum acceleration magnitude per high acceleration event
	:param max_lag: maximum allowed time between two high acceleration events
	:return: time difference or NaN if no such events where found
	"""
	left_timestamps = left[(left.acc0**2 + left.acc1**2 + left.acc2**2)**0.5 > min_acc_mag]
	right_timestamps = right[(right.acc0**2 + right.acc1**2 + right.acc2**2)**0.5 > min_acc_mag]
	for left_t in left_timestamps.index.values:
		for right_t in right_timestamps.index.values:
			if abs(left_t - right_t) < max_lag:
				return left_t - right_t, max(left_t, right_t)
	raise ValueError("Could not find a joint acceleration event with magnitude larger than %i "
					 "within a time frame of %.2f s" % (min_acc_mag, max_lag))

def get_time_diff(left_time_series, right_time_series):
	"""
	Check for missing sensor data e.g. caused by a silent disconnect of a Myo.

	:param left_time_series: one time series
	:param right_time_series: other time series
	:return: time difference between both time series lengths
	"""
	left_duration = left_time_series[-1] - left_time_series[0]
	right_duration = right_time_series[-1] - right_time_series[0]
	return abs(left_duration - right_duration)

def merge_interpolate_sensors(left, right, leftwards, method):
	"""
	Merge and interpolate two sensor data frames (sensor values are assumed to be integers).

	:param left: left sensors data frame
	:param right: right sensors data frame
	:param leftwards: if true merge leftwards (left indices) otherwise rightwards (right indices)
	:param method: interpolation method (c.f. pandas.Series.interpolate for available choices)
	:return: merged and interpolated sensor data
	"""
	# merge left and right and set the index on the merged column
	merged = pd.merge_ordered(left, right, on="time", suffixes=("_l", "_r")).set_index("time")

	# interpolate with the given method and reindex (resample) with the chosen indices
	interpolated = merged.interpolate(method=method)
	reindexed = interpolated.reindex(left.index if leftwards else right.index)

	# return cleaned data (drop NAN and round because sensor data is represented in integers)
	return reindexed.dropna().round()

def estimate_sync_params(left_imu, right_imu, sync, prefix="unknown"):
	"""
	Estimate the lag of both Myos by utilizing the IMU data of the clap sync phase and retrieve the
	synchronization time.

	:param left_imu: IMU data of the left Myo
	:param right_imu: IMU data of the right Myo
	:param sync: synchronization parameters
	:return lag: estimated lag between left and right Myo
	:return sync_time: time of the synchronization event
	"""
	if sync:
		lag, sync_time = find_joint_acc_lag(left_imu, right_imu, sync["min_acc_magnitude"], sync["max_lag"])
		LOG.debug("lag: %s", lag)
	else:
		lag = 0
		sync_time = None
		LOG.warning("%s: missing synchronisation data, assuming a shift of zero.", prefix)
	return lag, sync_time

def sync_two_myo_data(left, right, method, lag=0, prefix="unknown", max_time_diff=0.1):
	"""Merge and interpolate sensor data to match the Myo with the highest frequency.

	:param left: merged sensor data of the left Myo
	:param right: merged sensor data of the right Myo
	:param method: interpolation method (c.f. pandas.Series.interpolate for available choices)
	:param lag: the lag between left and right time series
	:max_time_diff: maximum allowed time difference between both sensor data
	:return: synchronized, merged and interpolated sensor data of two Myos
	"""
	# check that both Myos have sent about the same amount of sensor data
	time_diff = get_time_diff(left.index.values, right.index.values)
	sample_diff = len(left) - len(right)
	if time_diff > max_time_diff:
		LOG.error("%s: about %.2f s of sensor data are missing", prefix, time_diff)
		LOG.error("%s: number of samples differ by %s samples", prefix, sample_diff)

	# synchronize, merge and interpolate left and right
	right.index += lag
	leftwards = utils.get_frequency(left.index) > utils.get_frequency(right.index)
	return merge_interpolate_sensors(left, right, leftwards, method)
