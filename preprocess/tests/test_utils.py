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

import random

from preprocess import utils

import pandas as pd
import numpy as np

def test_ignore_channels_single():
	data_input = pd.DataFrame({
		"a": [1, 2, 3, 4],
		"b": [1, 2, 3, 4],
		"c": [1, 2, 3, 4],
	})
	pattern_input = ["a"]

	expected = pd.DataFrame({
		"b": [1, 2, 3, 4],
		"c": [1, 2, 3, 4],
	})
	result = utils.ignore_channels(data_input, pattern_input)

	pd.testing.assert_frame_equal(expected, result)

def test_ignore_channels_multi():
	data_input = pd.DataFrame({
		"a": [1, 2, 3, 4],
		"b": [1, 2, 3, 4],
		"c": [1, 2, 3, 4],
		"a_1": [1, 2, 3, 4],
	})
	pattern_input = ["a", "b"]

	expected = pd.DataFrame({
		"c": [1, 2, 3, 4],
	})
	result = utils.ignore_channels(data_input, pattern_input)

	pd.testing.assert_frame_equal(expected, result)

def test_ignore_channels_non_existent():
	data_input = pd.DataFrame({
		"a": [1, 2, 3, 4],
		"b": [1, 2, 3, 4],
		"c": [1, 2, 3, 4],
		"a_1": [1, 2, 3, 4],
	})
	pattern_input = ["a_1", "d"]

	expected = pd.DataFrame({
		"a": [1, 2, 3, 4],
		"b": [1, 2, 3, 4],
		"c": [1, 2, 3, 4],
	})
	result = utils.ignore_channels(data_input, pattern_input)

	pd.testing.assert_frame_equal(expected, result)



def mock_read_meta(prefix):
	return {"id": {"user": 5}, "common": {"typing_style": "touch_typing", "task_type": "uniform 3"}}

def test_file_filter_on_typist_all(monkeypatch):
	prefix = "no matter"
	user = None

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_typist(prefix, user)

	assert expected == result

def test_file_filter_on_typist_select(monkeypatch):
	prefix = "no matter"
	user = "5"

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_typist(prefix, user)

	assert expected == result

def test_file_filter_on_typist_select_int(monkeypatch):
	prefix = "no matter"
	user = 5

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_typist(prefix, user)

	assert expected == result

def test_file_filter_on_typist_select_not_in_filter(monkeypatch):
	prefix = "no matter"
	user = "6"

	expected = False
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_typist(prefix, user)

	assert expected == result

def test_file_filter_on_typist_select_style(monkeypatch):
	prefix = "no matter"
	user = "touch_typing"

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_typist(prefix, user)

	assert expected == result



def test_file_filter_on_task_type_all(monkeypatch):
	prefix = "no matter"
	task_type = []

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_task_type(prefix, task_type)

	assert expected == result

def test_file_filter_on_task_type_select(monkeypatch):
	prefix = "no matter"
	task_type = ["uniform", "text", "pangram", "game"]

	expected = True
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_task_type(prefix, task_type)

	assert expected == result

def test_file_filter_on_task_type_select_not_in_filter(monkeypatch):
	prefix = "no matter"
	task_type = ["text"]

	expected = False
	monkeypatch.setattr(utils, "read_meta", mock_read_meta)
	result = utils.file_filter_on_task_type(prefix, task_type)

	assert expected == result



def test_get_cut_indices_first():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [3, 3], "passwords_per_task": 2}
	}
	select = "first"

	expected = [1,3,6,10]
	result = utils.get_cut_indices(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_last():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [3, 3], "passwords_per_task": 2}
	}
	select = "last"

	expected = [5,6,11,15]
	result = utils.get_cut_indices(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_all():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [3, 3], "passwords_per_task": 2}
	}
	select = "all"

	expected = [1,3,2,6,5,6,6,10,9,12,11,15]
	result = utils.get_cut_indices(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_first_one_rep():
	data = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data.set_index("time", inplace=True)
	meta_data = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 3], "passwords_per_task": 2}
	}
	select = "first"

	expected = [1,3,2,6]
	result = utils.get_cut_indices(data, meta_data, select)

	assert expected == result

def test_get_cut_indices_last_one_rep():
	data = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data.set_index("time", inplace=True)
	meta_data = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 3], "passwords_per_task": 2}
	}
	select = "last"

	expected = [1,3,6,10]
	result = utils.get_cut_indices(data, meta_data, select)

	assert expected == result

def test_get_cut_indices_all_one_rep():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 3], "passwords_per_task": 2}
	}
	select = "all"

	expected = [1,3,2,6,5,6,6,10]
	result = utils.get_cut_indices(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_last_after_data():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [0.72, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [3, 3], "passwords_per_task": 2}
	}
	select = "last"

	expected = [5,6,11,14]
	result = utils.get_cut_indices(data_input, meta_data_input, select)

	assert expected == result



def test_get_cut_indices_between_first():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [],
		"cuts": [0.61, 0.79, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [2, 2], "passwords_per_task": 2}
	}
	select = "first"

	expected = [1,1,6,9]
	result = utils.get_cut_indices_between(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_between_last():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [],
		"cuts": [0.61, 0.79, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [2, 2], "passwords_per_task": 2}
	}
	select = "last"

	expected = [3,5,12,14]
	result = utils.get_cut_indices_between(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_between_all():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [],
		"cuts": [0.61, 0.79, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [2, 2], "passwords_per_task": 2}
	}
	select = "all"

	expected = [1,1,3,5,6,9,12,14]
	result = utils.get_cut_indices_between(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_between_first_one_rep():
	data = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data.set_index("time", inplace=True)
	meta_data = {
		"simple_cuts": [],
		"cuts": [0.64, 0.81, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 2], "passwords_per_task": 2}
	}
	select = "first"

	expected = [1,2,3,5]
	result = utils.get_cut_indices_between(data, meta_data, select)

	assert expected == result

def test_get_cut_indices_between_last_one_rep():
	data = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data.set_index("time", inplace=True)
	meta_data = {
		"simple_cuts": [],
		"cuts": [0.64, 0.81, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 2], "passwords_per_task": 2}
	}
	select = "last"

	expected = [1,2,6,9]
	result = utils.get_cut_indices_between(data, meta_data, select)

	assert expected == result

def test_get_cut_indices_between_all_one_rep():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [],
		"cuts": [0.64, 0.81, 0.82, 1.12, 1.2, 1.59],
		"task": {"attempts": [1, 2], "passwords_per_task": 2}
	}
	select = "all"

	expected = [1,2,3,5,6,9]
	result = utils.get_cut_indices_between(data_input, meta_data_input, select)

	assert expected == result

def test_get_cut_indices_between_last_after_data():
	data_input = pd.DataFrame({
		"time": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
		"arbitrary": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
	})
	data_input.set_index("time", inplace=True)
	meta_data_input = {
		"simple_cuts": [],
		"cuts": [0.7, 0.72, 0.82, 1.12, 1.2, 1.59, 1.79, 2.01],
		"task": {"attempts": [2, 2], "passwords_per_task": 2}
	}
	select = "last"

	expected = [3,5,12,14]
	result = utils.get_cut_indices_between(data_input, meta_data_input, select)

	assert expected == result



def test_slice_data_to_word_X():
	X_input = (
	np.array([
		[0.6, 0],
		[0.7, 0],
		[0.8, 0],
		[0.9, 0],
		[1.0, 0],
		[1.1, 0],
		[1.2, 0]
	]),
	np.array([
		[1.3, 0],
		[1.4, 0],
		[1.5, 0],
		[1.6, 0],
		[1.7, 0],
		[1.8, 0],
		[1.9, 0]
	]))
	cuts_input = (
		[0, 2, 3, 6],
		[1, 5]
	)

	expected = (
	[
		[0.6, 0],
		[0.7, 0],
		[0.8, 0]
	],
	[
		[0.9, 0],
		[1.0, 0],
		[1.1, 0],
		[1.2, 0]
	],
	[
		[1.4, 0],
		[1.5, 0],
		[1.6, 0],
		[1.7, 0],
		[1.8, 0]
	])
	result = utils.slice_data_to_word(X_input, cuts_input)
	result_without_meta = result[0]

	np.testing.assert_array_equal(expected[0], result_without_meta[0])
	np.testing.assert_array_equal(expected[1], result_without_meta[1])
	np.testing.assert_array_equal(expected[2], result_without_meta[2])

def test_slice_data_to_word_y():
	y_input = (
		[0, 1, 0, 0, 1, 1, 0],
		[0, 0, 1, 1, 1, 0, 0]
	)
	cuts_input = (
		[0, 2, 3, 6],
		[1, 5]
	)

	expected = (
		[0, 1, 0],
		[0, 1, 1, 0],
		[0, 1, 1, 1, 0]
	)
	result = utils.slice_data_to_word(y_input, cuts_input)
	result_without_meta = result[0]

	assert expected == result_without_meta

def test_slice_data_to_word_y_with_meta():
	y_input = (
		[0, 1, 0, 0, 1, 1, 0],
		[0, 0, 1, 1, 1, 0, 0]
	)
	cuts_input = (
		[0, 2, 3, 6],
		[1, 5]
	)
	meta_input = ("a", "b")

	expected = (
		(
			[0, 1, 0],
			[0, 1, 1, 0],
			[0, 1, 1, 1, 0]
		),
		("a", "a", "b"),
	)
	result = utils.slice_data_to_word(y_input, cuts_input, meta_input)
	result_without_meta = result[0]

	assert expected == result

def test_get_frequency():
	for i in range(1000):
		start, step = random.randint(0, 1000), random.randint(1, 1000)
		end = start + random.randint(2, 1000) * step
		signal = np.arange(start, end, step)
		assert utils.get_frequency(signal) == 1/step
