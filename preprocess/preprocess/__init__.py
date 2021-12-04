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
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
from joblib import Memory

from . import utils, fuse, encode, filt

LOG = logging.getLogger(__name__)
MEM = Memory("cache", verbose=0)

def single_task(prefix, method="index", rescale_imu=False, cut=True, ignore=[],
				emg_pli_filter=False, emg_hp_filter=True,
				input_dtype="float32", encoding="binary", target="press",
				freq=200, select=None, add_shifted_samples=False):
	"""
	Load and preprocess the sensor and key data of a single data taking.

	:param prefix: prefix used for a single data taking
	:param method: method to be used for interpolating IMU and/or EMG data
	               (c.f. pandas.Series.interpolate for available choices)
	:param rescale_imu: whether or not to rescale the IMU sensor values to meaningful units
	:param cut: whether or not to cut the sensor data to the actual task length
	:param ignore: a list of data channels to ignore
	:param emg_pli_filter: whether or not to filter PLI noise from raw EMG data
	:param emg_hp_filter: whether or not to filter low frequencies from the raw EMG data
	:param input_dtype: dtype string or dictionary of 'column name': 'dtype string' for input vector
	:param encoding: key data encoding ('binary', 'finger', 'multiclass', 'multiclass_alpha' or 'shift')
	:param target: target (either 'state', 'press' events or 'release' events) to be encoded
	:param freq: the target frequency of the sensor and key data
	:param select: (test-data only) select which passwords should be returned ('first', 'last' or 'all')
	:param add_shifted_samples: whether to add shifted truth values to a multiclass training
	:return: preprocessed sensor data and encoded/sampled key data of a single task, and the
	         meta data
	"""
	# load the raw data (key, meta and sensor data with corrected timestamps)
	key_data = utils.read_csv(prefix + ".key.csv", index_col=0)
	left_emg, left_imu = utils.read_myo(prefix, "left", rescale_imu, dtype=input_dtype)
	right_emg, right_imu = utils.read_myo(prefix, "right", rescale_imu, dtype=input_dtype)
	meta_data = utils.read_meta(prefix, rescale_imu)

	# estimate time lag based on raw IMU data and return the time of the synchronization event
	lag, sync_time = fuse.estimate_sync_params(left_imu, right_imu, meta_data["sync"], prefix)

	# filter EMG data and ensure that the dtype does not change
	pli_freq = meta_data["common"]["power_line_frequency"]
	emg_filters = [
		partial(filt.notch, f0=pli_freq) if emg_pli_filter else None,
		filt.highpass if emg_hp_filter else None,
	]
	left_emg = filt.apply_filters(filter(None, emg_filters), left_emg).astype(input_dtype)
	right_emg = filt.apply_filters(filter(None, emg_filters), right_emg).astype(input_dtype)

	# resample sensor data to a common frequency in Hz
	if freq is not None:
		left_emg = utils.resample(left_emg, freq, method)
		left_imu = utils.resample(left_imu, freq, method)
		right_emg = utils.resample(right_emg, freq, method)
		right_imu = utils.resample(right_imu, freq, method)

	# fuse EMG and IMU sensor data of both Myos
	left = fuse.merge_interpolate_sensors(left_emg, left_imu, True, method)
	right = fuse.merge_interpolate_sensors(right_emg, right_imu, True, method)

	# fuse both Myos
	fused_sensors = fuse.sync_two_myo_data(left, right, method, lag, prefix)

	# drop channels listed as ignore
	if ignore:
		fused_sensors = utils.ignore_channels(fused_sensors, ignore)

	# cut away the first part of the data that does not include the task
	if cut:
		fused_sensors = utils.cut_to_task(key_data, fused_sensors, meta_data, sync_time)

	# encode and resample the key data based on the sensor timestamps
	sampled_keys, n_classes = encode.resample(key_data, fused_sensors.index, encoding, target, add_shifted_samples)

	# if the password data is used, find the indices to later cut out the relevant data
	cut_indices = []
	if "cuts" in meta_data and select is not None:
		cut_indices = utils.get_cut_indices_between(fused_sensors, meta_data, select)

	return fused_sensors, sampled_keys, cut_indices, n_classes, meta_data

def single_task_np(*args, **kwargs):
	"""
	Load and preprocess the sensor and key data of a single data taking.
	"""
	X, y, cuts, n_classes, meta_data = single_task(*args, **kwargs)
	# ensure that labels are immutable ==> copy.deepcopy will not create a new object
	X_labels = tuple(X.columns.tolist())
	if y.shape[-1] == 1:
		return X.values, y.values.ravel(), cuts, X_labels, n_classes, meta_data
	else:
		return X.values, y.values, cuts, X_labels, n_classes, meta_data

@MEM.cache
def all_tasks_np(path, file_filter=None, **kwargs):
	"""
	Load and preprocess the sensor and key data of every task sufficing to the path and file_filter.

	:param path: path to search tasks in
	:param file_filter: predicate for determining which tasks to choose
	:param kwargs: keyword arguments to be forwarded to the single_task function
	:return: list of numpy arrays of preprocessed sensor data and encoded/sampled key data along
	         with a list of their corresponding column labels, and a dataframe of all identifiers
	         for applying grouped cross-validation
	"""
	# determine files to be loaded
	args = sorted(utils.get_task_prefices(path, file_filter))
	if not args:
		raise ValueError(f"No files selected: Wrong {path=} or invalid {file_filter=}")
	func = partial(single_task_np, **kwargs)

	# load all files in parallel and put them together
	with multiprocessing.Pool() as pool:
		X, y, cuts, X_labels, n_classes, meta_data = zip(*pool.map(func, args))
	X_labels, n_classes = X_labels[0], n_classes[0]

	# cut the file data into single slices of (pass)word data, replicate meta data for each slice
	if cuts[0] != []:
		X, _ = utils.slice_data_to_word(X, cuts)
		y, meta_data = utils.slice_data_to_word(y, cuts, meta_data)
	assert(len(X) == len(y) == len(meta_data))

	# determine unique classes without counting sentinel values (e.g. -1)
	classes = []
	for sub_y in y:
		sub_classes = np.unique(sub_y)
		classes.append(sub_classes[sub_classes >= 0])
	# test if n_classes is correct and raise a warning if not
	real_n_classes = len(np.unique(np.concatenate(classes)))
	real_n_classes = 1 if real_n_classes == 2 else real_n_classes
	if n_classes != real_n_classes:
		LOG.warning("Expected %s classes, got %s instead", n_classes, real_n_classes)

	# log information about the data
	LOG.info("Got %s timeseries in X (shape of first series: %s)", len(X), X[0].shape)
	LOG.info("Got %s timeseries in y (shape of first series: %s)", len(y), y[0].shape)
	LOG.info("X_labels: %s", X_labels)
	LOG.info("n_classes: %s", n_classes)
	LOG.info("unique labels: %s", dict(zip(*np.unique(y[0], return_counts=True))))
	identifiers = pd.DataFrame([item["id"] for item in meta_data])
	for id_type in identifiers:
		LOG.info(
			"unique identifier['%s']: %s",
			id_type,
			dict(zip(*np.unique(identifiers[id_type], return_counts=True)))
		)

	return X, y, X_labels, n_classes, meta_data

def single_task_pd(*args, **kwargs):
	"""
	Load and preprocess the sensor and key data of a single data taking and merge it into a
	pandas DataFrame.
	"""
	fused_sensors, sampled_keys, _, n_classes, meta_data = single_task(*args, **kwargs)
	df = pd.merge(fused_sensors, sampled_keys, on="time")
	df["user"] = meta_data["id"]["user"]
	df["collection"] = meta_data["id"]["collection"]
	df["typing_style"] = meta_data["common"]["typing_style"]
	df["task_type"] = meta_data["common"]["task_type"]
	df["task_id"] = meta_data["common"]["task_id"]
	df["keyboard_layout"] = meta_data["keyboard"]["layout"]
	df["n_classes"] = n_classes
	return df

@MEM.cache
def all_tasks_pd(path, file_filter=None, **kwargs):
	# determine files to be loaded
	args = sorted(utils.get_task_prefices(path, file_filter))
	if not args:
		raise ValueError(f"No files selected: Wrong {path=} or invalid {file_filter=}")
	func = partial(single_task_pd, **kwargs)

	# load all files in parallel and put them together
	with multiprocessing.Pool() as pool:
		dfs = pool.map(func, args)
	df = pd.concat(dfs)
	df = df.reset_index()
	return df
