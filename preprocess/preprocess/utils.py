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
import functools
import json
import pathlib

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

LOG = logging.getLogger(__name__)

################################### read raw data functionalities ##################################

ACCELEROMETER_SCALE = 2048 / 9.81
GYROSCOPE_SCALE = 16
ORIENTATION_SCALE = 16384

def read_csv(filename, **kwargs):
	return pd.read_csv(filename, engine="c", **kwargs)

@functools.lru_cache()
def read_meta(prefix, rescale_imu=False):
	with open(prefix + ".meta.json") as file_obj:
		result = json.load(file_obj)
	if rescale_imu and result["sync"]:
		result["sync"]["min_acc_magnitude"] /= ACCELEROMETER_SCALE
	return result

def read_sensor(file_name, rescale_imu=False, usecols=None, dtype="float32"):
	"""
	Load a single type of sensor data of a single Myo of a single data taking.

	:param file_name: the file name of the sensor data in csv format
	:param rescale_imu: whether or not to rescale the IMU sensor values to meaningful units
	:param usecols: list or predicate for which columns to read in
	:param dtype: dtype string or dictionary of 'column name': 'dtype string'
	:return: (optionally rescaled) sensor data
	"""
	sensor_data = read_csv(file_name, usecols=usecols, dtype=dtype)
	sensor_data["time"] = equidistant(sensor_data["time"].values)
	sensor_data = sensor_data.set_index("time")
	if rescale_imu:
		sensor_data[["acc0", "acc1", "acc2"]] /= ACCELEROMETER_SCALE
		sensor_data[["gyro0", "gyro1", "gyro2"]] /= GYROSCOPE_SCALE
		sensor_data[["quat0", "quat1", "quat2", "quat3"]] /= ORIENTATION_SCALE
	return sensor_data

def check_packet_loss(characteristics):
	"""
	Check for the correct order of Bluetooth characteristics to detect packet loss.

	:param characteristics: list of Bluetooth characteristics
	:return: whether or not the order of Bluetooth characteristics is as expected
	"""
	length = len(characteristics)
	expected = np.tile([0, 0, 1, 1, 2, 2, 3, 3], length // 8 + 1)
	return not np.array_equal(characteristics, expected[:length])

def read_myo(prefix, subprefix, rescale_imu=None, dtype="float32"):
	"""
	Load EMG and IMU data of a single Myo of a single data taking. Note that a warning is emitted on
	detecting potentially missing EMG data.

	:param prefix: common filename prefix
	:param subprefix: subprefix (e.g. the arm) of the Myo data files
	:param rescale_imu: whether or not to rescale the IMU sensor values to meaningful units
	:param dtype: dtype string or dictionary of 'column name': 'dtype string'
	:return: EMG and IMU data
	"""
	# load EMG and IMU data
	emg_data = read_sensor(".".join([prefix, subprefix, "emg.csv"]), usecols=lambda x: x not in ["moving"], dtype=dtype)
	imu_data = read_sensor(".".join([prefix, subprefix, "imu.csv"]), rescale_imu=rescale_imu, dtype=dtype)

	# check for packet loss (only possible for EMG data)
	if check_packet_loss(emg_data["characteristics"]):
		LOG.warning("%s.%s.emg.csv: At least one packet loss", prefix, subprefix)

	# return EMG and IMU data without the "characteristics" column
	emg_data.drop(["characteristics"], axis=1, inplace=True)
	return emg_data, imu_data

def get_task_prefices(path, file_filter=None):
	"""
	Get all task specific prefices given a path and file_filter predicate.

	:param path: path to search task prefices in
	:param file_filter: predicate for determining which task prefix to choose
	:return: list of task prefices
	"""
	# determine files to be loaded
	prefices = []
	for meta_file in pathlib.Path(path).glob("*.meta.json"):
		prefix = str(meta_file).replace(".meta.json", "")
		if file_filter and not file_filter(prefix):
			LOG.debug("Exclude prefix: %s", prefix)
			continue
		prefices.append(prefix)
	return prefices

################################### time related functionalities ###################################

def get_frequency(time_series):
	"""Return the sample frequency of the given time series."""
	return (len(time_series) - 1) / (time_series[-1] - time_series[0])

def equidistant(timestamps):
	"""Return linearly spaced timestamps based on the given (arbitrarily spaced) timestamps."""
	return np.linspace(timestamps[0], timestamps[-1], len(timestamps))

def resample(df, freq, assume_sorted=True, method="linear"):
	"""Resample and interpolate a pandas dataframe without a datetime-like index."""
	prev_freq = get_frequency(df.index)
	if prev_freq > freq:
		LOG.warning("Downsampling data from %s to %s Hz", prev_freq, freq)
	steps = int((df.index.values[-1] - df.index.values[0]) * freq) + 1
	new_timestamps = df.index.values[0] + np.linspace(0, (steps - 1) / freq, steps)
	interpolator = interp1d(df.index, df, axis=0, assume_sorted=assume_sorted, bounds_error=False, kind=method)
	new_df = pd.DataFrame(interpolator(new_timestamps), columns=df.columns)
	new_df["time"] = new_timestamps
	new_df = new_df.set_index("time").dropna().astype(df.dtypes)
	return new_df

def cut_to_task(key_data, sensor_data, meta_data, sync_time):
	"""Cut the given sensor data to the task."""
	return_presses = key_data.loc[(key_data["keycode"] == 36) & (key_data["event"] == "press")]
	# check for missing sync or missing return
	if not sync_time or any([string.startswith("SYNC_ERROR_B") for string in meta_data["notes"]]):
		# time from start to first return
		start = sensor_data.index.values[0]
		end = return_presses.index.values[0]
	else:
		# time from first to second return
		start = return_presses.loc[return_presses.index >= sync_time].index.values[0]
		end = return_presses.loc[return_presses.index >= sync_time].index.values[1]

	# cut between start and end
	#cut = start + (end - start)/2
	# set cut to at least 400 ms before the second return (or start)
	#if cut > end - 0.4:
	cut = max(end - 0.4, start)
	LOG.debug("cut before time: %s", cut)

	# cut the data
	return sensor_data.truncate(before=cut)

############################################ data filter ###########################################

def file_filter_on_typist(prefix, user=None):
	"""
	File filter to select data based on user IDs or common typing styles.

	:param prefix: file prefix for which to apply this predicate function
	:param user: an ID (string or integer), a typing style (string) or None to consider all users
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	if user is None:
		return True
	if user == "min":
		return file_filter_min(prefix)
	if user == "min_group":
		return file_filter_min_group(prefix)
	if user == "three_users":
		return file_filter_three_users(prefix)
	if user == "unknown":
		return file_filter_unknown_user(prefix)
	if user == "known":
		return file_filter_known_user(prefix)
	if user == "de" or user == "en":
		return file_filter_on_keyboard_layout(prefix, user)
	else:
		try:
			return meta_data["id"]["user"] == int(user)
		except ValueError:
			return meta_data["common"]["typing_style"] == user

def file_filter_on_task_type(prefix, task_types=[]):
	"""
	File filter to select data based on a list of task types.

	:param prefix: file prefix for which to apply this predicate function
	:param task_types: a list of required task types, an empty list will accept every task type
					   (note: task types must not be fully specified, a common root is sufficient)
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	if not task_types:
		return True
	else:
		for task_type in task_types:
			if task_type == "unknown":
				return file_filter_unknown_data(prefix)
			if task_type == "known":
				return file_filter_known_data(prefix)
			if task_type in meta_data["common"]["task_type"]:
				break
		else:
			return False
		return True

def file_filter_on_keyboard_layout(prefix, layout):
	"""
	File filter to select data based on the keyboard layout.

	:param prefix: file prefix for which to apply this predicate function
	:param layout: the keyboard layout (us/de)
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	if meta_data["keyboard"]["layout"] == layout:
		return True
	return False

def file_filter_min(prefix):
	"""
	File filter to select a predefined min sample (user 5, 'unform 5' and 'uniform 2').

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	task_type = meta_data["common"]["task_type"]
	user = meta_data["id"]["user"]
	return user == 5 and (task_type in ("uniform 5", "uniform 2"))

def file_filter_unknown_user(prefix):
	"""
	File filter to select unknown users in the test data.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	notes = meta_data["notes"]
	return not any("USER_IN_TRAIN_DATA" in note for note in meta_data["notes"])

def file_filter_known_user(prefix):
	"""
	File filter to select known users in the test data.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	notes = meta_data["notes"]
	return any("USER_IN_TRAIN_DATA" in note for note in meta_data["notes"])

def file_filter_unknown_data(prefix):
	"""
	File filter to select only unknown data, i.e. passwords typed by only
	one participant.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	task_id = meta_data["common"]["task_id"]
	return task_id in [8, 9, 10, 11]

def file_filter_known_data(prefix):
	"""
	File filter to select known data, i.e. passwords typed by multiple
	participants.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	task_id = meta_data["common"]["task_id"]
	return task_id in [0, 1, 2, 3, 4, 5, 6, 7]

def file_filter_min_group(prefix):
	"""
	File filter to select a predefined min sample from multiple users.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	task_type = meta_data["common"]["task_type"]
	user = meta_data["id"]["user"]
	return user in range(5) and (task_type in ("uniform 2",))

def file_filter_three_users(prefix):
	"""
	File filter to select the first three users.

	:param prefix: file prefix for which to apply this predicate function
	:return: whether the prefix satisfies the required conditions or not
	"""
	meta_data = read_meta(prefix)
	task_type = meta_data["common"]["task_type"]
	user = meta_data["id"]["user"]
	return user in range(3)

class FileFilter():
	"""
	File filter to select data based on user(s) and task type(s)

	:param user: a number or a string chosen from {'hybrid', 'touch_typing', 'min', ...}
	:param task_types: a string or list of strings chosen from {'text', 'uniform', 'game', ...}
	"""
	def __init__(self, user, task_types):
		self.user = user
		self.task_types = task_types

	def __call__(self, prefix):
		"""
		Apply this filter on the chosen prefix.

		:param prefix: file prefix for which to apply this predicate function
		:return: whether the prefix satisfies the required conditions or not
		"""
		return file_filter_on_typist(prefix, self.user) and file_filter_on_task_type(prefix, self.task_types)

	def __repr__(self):
		return "%s(user=%s, task_types=%s)" % (self.__class__.__name__, self.user, self.task_types)


######################################## split password data #######################################

def get_cut_indices(data, meta_data, select):
	"""
	Get the indices between which to cut the data to receive single chunks of password-only data.
	The given cuts are assumed to mark the beginning and end of the passwords recordings, as well as
	the points of time in between two password recordings.
	Works with password recordings only, as it heavily relies on certain meta data information.

	:param data: the data based on which to get the cut indices
	:param meta_data: the meta data containing cut timestamps, number of passwords typed and
					  attempts
	:param select: whether to cut out the first of repeatedly typed passwords (unknown password to
				 the user) or the last (known password)
	:return: an array of the cut indices alternating start and end index of cuts, i.e.
			 [start, end, start...]
	"""
	cut_indices = []
	cuts = meta_data["simple_cuts"]
	num_samples = meta_data["task"]["passwords_per_task"]
	change = 0
	# untrained pw entries
	if select == "first":
		# for the number of passwords
		for sample in range(num_samples):
			# cut between cuts[change] and cuts[change+1]
			cut_indices.append(data.index.get_loc(cuts[change], method="ffill"))
			cut_indices.append(data.index.get_loc(cuts[change+1], method="bfill"))
			change += meta_data["task"]["attempts"][sample]
	# trained pw entries
	elif select == "last":
		# for the number of passwords
		for sample in range(num_samples):
			change += meta_data["task"]["attempts"][sample]
			# cut between cuts[change-1] and cuts[change]
			cut_indices.append(data.index.get_loc(cuts[change-1], method="ffill"))
			try:
				cut_indices.append(data.index.get_loc(cuts[change], method="bfill"))
			except KeyError: # if the cut occurs after the data timestamps end
				cut_indices.append(len(data.index) - 1)
	elif select == "all":
		# for the number of cuts
		for sample in range(len(cuts)-1):
			# use every cut
			cut_indices.append(data.index.get_loc(cuts[sample], method="ffill"))
			try:
				cut_indices.append(data.index.get_loc(cuts[sample+1], method="bfill"))
			except KeyError: # if the cut occurs after the data timestamps end
				cut_indices.append(len(data.index) - 1)
	else:
		raise ValueError(f"Error: {select=} is not a valid choice.")
	return cut_indices

def get_cut_indices_between(data, meta_data, select):
	"""
	Get the indices between which to cut the data to receive single chunks of password-only data.
	The given cuts are assumed to mark the beginning and end of each password separately.
	Works with password recordings only, as it heavily relies on certain meta data information.

	:param data: the data based on which to get the cut indices
	:param meta_data: the meta data containing cut timestamps, number of passwords typed and
					  attempts
	:param select: whether to cut out the first of repeatedly typed passwords (unknown password to
				 the user) or the last (known password)
	:return: an array of the cut indices alternating start and end index of cuts, i.e.
			 [start, end, start...]
	"""
	cut_indices = []
	cuts = meta_data["cuts"]
	simple_cuts = meta_data["simple_cuts"]
	num_samples = meta_data["task"]["passwords_per_task"]
	attempts = meta_data["task"]["attempts"]
	# find gaps in cuts due to no typing and reduce attempts accordingly
	if len(cuts) != sum(attempts) * 2:
		LOG.warning("Attempts do not match for record %s (task %s)!", meta_data["id"]["collection"], meta_data["common"]["task_id"])
		LOG.warning("Attempts before change: %s.", attempts)
		offset = 0
		for sample in range(num_samples):
			num_missing_cuts = 0
			for attempt in range(attempts[sample]):
				# if the current cut is larger than the next simple cut there is a gap in the cuts
				if cuts[(offset + attempt - num_missing_cuts)*2] > simple_cuts[offset + attempt + 1]:
					num_missing_cuts += 1
			# correct the number of attempts for the current password sample
			attempts[sample] -= num_missing_cuts
			offset += attempts[sample]
		LOG.warning("Attempts after auto-correction: %s.", attempts)
	# untrained pw entries
	change = 0
	if select == "first":
		# for the number of passwords
		for sample in range(num_samples):
			# cut between cuts[change] and cuts[change+1]
			cut_indices.append(data.index.get_loc(cuts[change], method="bfill"))
			cut_indices.append(data.index.get_loc(cuts[change+1], method="ffill"))
			change += attempts[sample] * 2 # * 2 because of two entries per pw
	# trained pw entries
	elif select == "last":
		# for the number of passwords
		for sample in range(num_samples):
			change += attempts[sample] * 2
			# cut between cuts[change-1] and cuts[change]
			cut_indices.append(data.index.get_loc(cuts[change-2], method="bfill"))
			try:
				cut_indices.append(data.index.get_loc(cuts[change-1], method="ffill"))
			except KeyError:
				cut_indices.append(len(data.index) - 1) # if the cut occurs after the data timestamps end
	elif select == "all":
		# for the number of cuts, add each cut pair
		for sample in range(0, len(cuts)-1, 2):
			# use every cut
			cut_indices.append(data.index.get_loc(cuts[sample], method="bfill"))
			cut_indices.append(data.index.get_loc(cuts[sample+1], method="ffill"))
	else:
		raise ValueError(f"Error: {select=} is not a valid choice.")
	return cut_indices

def slice_data_to_word(data, cuts, meta_data=None):
	data_temp = []
	meta_temp = []
	# for the data extracted from each file...
	for i in range(len(data)):
		current_cut_array = cuts[i]
		# cut out single words
		for c in range(0, len(current_cut_array)-1, 2):
			data_temp.append(data[i][current_cut_array[c]:current_cut_array[c+1]+1])
			if meta_data:
				meta_temp.append(meta_data[i])
	return tuple(data_temp), tuple(meta_temp)

############################################### other ##############################################

def ignore_channels(data, patterns=[]):
	"""
	Drop columns using a list of patterns.

	:param patterns: list of string patterns (data column headers) to search for
	:return: the given data without the columns matching the given patterns
	"""
	ignored = set([])
	for string in patterns:
		for channel in data.columns:
			if string in channel:
				ignored.add(channel)
	LOG.debug("Dropping the following channels: %s", sorted(ignored))
	return data.drop(ignored, axis=1)

def create_tag(user, task_types, encoding, target, ignored_channels):
	"""
	Determine a descriptive tag based on user and task selection, combined with the selected sensor
	channels and target encoding.

	:param user: a string representation of which users data is used to fit the estimator
	:param task_type: a list of task types to be used to fit the estimator
	:param encoding: key data encoding ('binary', 'finger', 'multiclass', 'multiclass_alpha' or 'shift')
	:param target: target (eithter 'state', 'press' events or 'release' events) to be encoded
	:param ignored_channels: the ignored sensor channels
	:return: string representation of the data configuration
	"""
	if not user:
		user = "all"
	elif user not in ("touch_typing", "hybrid", "min", "min_group", "three_users", "unknown", "known", "en", "de"):
		user = "within_user_" + str(user)
	if not task_types:
		task_types = "gptu"
	else:
		task_types = "".join(sorted(set(task[0] for task in task_types)))
	channels = "".join(
		channel[0] for channel in ["emg", "acc", "gyro", "quat"]
		if channel not in ignored_channels
	)
	return "_".join([user, task_types, encoding, target, channels])
