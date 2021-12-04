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

import pytest
import pandas as pd
import numpy as np

import preprocess.encode

# the unique list of keycodes from the data study (union of English and German keyboard layouts)
ALL_KEYCODES = [
	10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, # number row, =/+ (US) and backspace
	24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, # upper row and enter
	38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51,     # home row and #/' (DE)
	50, 94, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, # shift_l, </> (DE), lower row and shift_r
	65,                                                 # space
]

def test_keycodes():
	all_keycodes = list(preprocess.encode.KEYMAP_DE.keys()) + list(preprocess.encode.KEYMAP_US.keys())
	assert sorted(ALL_KEYCODES) == sorted(set(all_keycodes))



def test_remove_auto_press():
	key_input = pd.DataFrame([(27, "r", "press"), (27, "r", "press"), (27, "r", "release"), (38, "a", "press"), (38, "a", "release")])

	expected = pd.DataFrame([(27, "r", "press"), (27, "r", "release"), (38, "a", "press"), (38, "a", "release")], index=[0,2,3,4], dtype=object)
	result = preprocess.encode.remove_auto_press(key_input)

	pd.testing.assert_frame_equal(expected, result)



################# key identification (multiclass) ##################################################

def test_encode_id_repeat_key():
	key_input = pd.DataFrame([(27, "r", "press"), (27, "r", "press"), (27, "r", "release"), (38, "a", "press"), (38, "a", "release")])

	expected = pd.Series([27, 27, 0, 38, 0])
	result = preprocess.encode.encode_id(key_input)

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_index():
	key_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	key_input.set_index("time", inplace=True)

	expected = pd.Series([27, 0, 38, 0, 27, 0, 38], index=key_input.index)
	result = preprocess.encode.encode_id(key_input)

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_overlap_half():
	key_input = pd.DataFrame([(27, "r", "press"), (38, "a", "press"), (27, "r", "release"), (38, "a", "release")])

	expected = pd.Series([27, 38, 38, 0])
	result = preprocess.encode.encode_id(key_input)

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_overlap_full():
	key_input = pd.DataFrame([(27, "r", "press"), (38, "a", "press"), (38, "a", "release"), (27, "r", "release")])

	expected = pd.Series([27, 38, 27, 0])
	result = preprocess.encode.encode_id(key_input)

	pd.testing.assert_series_equal(expected, result)



def test_encode_id_single_repeat_key_press():
	key_input = pd.DataFrame([(27, "r", "press"), (27, "r", "press"), (27, "r", "release"), (38, "a", "press"), (38, "a", "release")], columns=["keycode", "keysym", "event"])

	expected = pd.Series([27, 27, 0, 38, 0])
	result = preprocess.encode.encode_single_id(key_input, "press")

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_single_repeat_key_release():
	key_input = pd.DataFrame([(27, "r", "press"), (27, "r", "press"), (27, "r", "release"), (38, "a", "press"), (38, "a", "release")], columns=["keycode", "keysym", "event"])

	expected = pd.Series([0, 0, 27, 0, 38])
	result = preprocess.encode.encode_single_id(key_input, "release")

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_single_index_press():
	key_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	key_input.set_index("time", inplace=True)

	expected = pd.Series([27, 0, 38, 0, 27, 0, 38], index=key_input.index)
	result = preprocess.encode.encode_single_id(key_input, "press")

	pd.testing.assert_series_equal(expected, result)

def test_encode_id_single_index_release():
	key_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	key_input.set_index("time", inplace=True)

	expected = pd.Series([0, 27, 0, 38, 0, 27, 0], index=key_input.index)
	result = preprocess.encode.encode_single_id(key_input, "release")

	pd.testing.assert_series_equal(expected, result)



################# key identification (multilabel) ##################################################

def test_create_zero_initialized_df():
	row_index_input = pd.RangeIndex(stop=2)
	columns_input = [27, 38, 65]

	expected = pd.DataFrame({
		27: [0.0, 0.0],
		38: [0.0, 0.0],
		65: [0.0, 0.0]
	}, dtype="float32")
	result = preprocess.encode.create_zero_initialized_df(row_index_input, columns_input)

	pd.testing.assert_frame_equal(expected, result)

def test_create_zero_initialized_df_index():
	row_index_input = [0.61, 0.7]
	columns_input = [27, 38, 65]

	expected = pd.DataFrame({
		27: [0.0, 0.0],
		38: [0.0, 0.0],
		65: [0.0, 0.0]
	}, dtype="float32")
	expected.index = row_index_input
	result = preprocess.encode.create_zero_initialized_df(row_index_input, columns_input)

	pd.testing.assert_frame_equal(expected, result)



@pytest.mark.skip(reason="currently no multilabel approach in use and this fails with float32")
def test_encode_as_columns():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	keycodes_input = [27, 38, 65]

	expected = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"],
		27: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
		38: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
		65: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.encode_as_columns(raw_key_events_input, keycodes_input)

	pd.testing.assert_frame_equal(expected, result)

@pytest.mark.skip(reason="currently no multilabel approach in use and this fails with float32")
def test_encode_as_columns_overlap():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [39, 39, 38, 38, 39, 27, 27],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "press", "press", "release", "release", "press", "release"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	keycodes_input = [27, 38, 39]

	expected = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [39, 39, 38, 38, 39, 27, 27],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "press", "press", "release", "release", "press", "release"],
		27: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
		38: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		39: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.encode_as_columns(raw_key_events_input, keycodes_input)

	pd.testing.assert_frame_equal(expected, result)

@pytest.mark.skip(reason="currently no multilabel approach in use and this fails with float32")
def test_encode_as_columns_unconsidered_key():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 21, 21, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	keycodes_input = [27, 38, 65]

	expected = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 21, 21, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"],
		27: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		38: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
		65: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.encode_as_columns(raw_key_events_input, keycodes_input)

	pd.testing.assert_frame_equal(expected, result)



def test_add_shift_for_identification():
	keys_input = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1.0, -1.0, -1.0, 26.0, -1.0, -1.0, 16.0, -1.0, -1.0, 26.0, -1.0]
	})
	max_shift_input = 2

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1.0, -1.0, 26.0, 26.0, 26.0, 16.0, 16.0, 16.0, 26.0, 26.0, -1.0]
	})

	result = preprocess.encode.add_shift_for_identification(keys_input, max_shift_input)
	pd.testing.assert_frame_equal(expected, result)

def test_add_shift_for_identification_overlapping_shift():
	keys_input = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, -1, 26, -1, -1, 16, -1, -1, 26, -1]
	})
	max_shift_input = 10 # default

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, 26, 26, 26, 16, 16, 16, 26, 26, -1]
	})

	result = preprocess.encode.add_shift_for_identification(keys_input, max_shift_input)
	pd.testing.assert_frame_equal(expected, result)

def test_add_shift_for_identification_non_overlapping_shift():
	keys_input = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, 26, -1, -1, -1, 16, -1, -1, 26, -1]
	})
	max_shift_input = 1

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, 26, 26, 26, -1, 16, 16, 16, 26, 26, -1]
	})

	result = preprocess.encode.add_shift_for_identification(keys_input, max_shift_input)
	pd.testing.assert_frame_equal(expected, result)

def test_add_shift_for_identification_last_is_key():
	keys_input = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, 26, -1, -1, -1, 16, -1, -1, -1, 26]
	})
	max_shift_input = 1

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, 26, 26, 26, -1, 16, 16, 16, -1, 26, 26]
	})

	result = preprocess.encode.add_shift_for_identification(keys_input, max_shift_input)
	pd.testing.assert_frame_equal(expected, result)



################# resample #########################################################################

def test_resample_detection():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "binary"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		#"keycode": [0, 0, 27, 38, 38, 38, 27, 27, 27],
		#"keysym": [0, 0, "", "", "", "", "", "", ""],
		#"event": [0, 0, "release", "press", "release", "release", "press", "press", "release"],
		"y": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
	}, dtype="float32")
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 1 == result[1]

def test_resample_detection_index():
	raw_key_events_input = pd.DataFrame({
		"time": [0.4, 0.8],
		"keycode": [27, 27],
		"keysym": ["", ""],
		"event": ["press", "release"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], name="time")
	encoding_input = "binary"

	expected = pd.DataFrame({
		"time": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
		"y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
	}, dtype="float32")
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 1 == result[1]

def test_resample_detection_encode_press():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.39, 1.41, 1.6],
		"keycode": [27, 27, 38, 38, 27, 27, 38, 38, 36],
		"keysym": ["", "", "", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name="time")
	encoding_input = "binary"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
	}, dtype="float32")
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, target="press")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 1 == result[1]

def test_resample_identification():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"y": [-1, -1, -1, 26, -1, -1, 16, 16, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_collision():
	raw_key_events_input = pd.DataFrame({
		"time": [0.59, 0.6, 0.66, 0.7, 1.1, 1.21, 1.5],
		"keycode": [27, 38, 27, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "press", "release", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"y": [-1, 26, -1, -1, -1, -1, 16, 16, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_collision():
	raw_key_events_input = pd.DataFrame({
		"time": [0.59, 0.6, 0.66, 0.7, 1.1, 1.21, 1.5],
		"keycode": [38, 27, 27, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "press", "release", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"y": [-1, 16, -1, -1, -1, -1, 16, 16, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_encode_press():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.39, 1.41, 1.6],
		"keycode": [27, 27, 38, 38, 27, 27, 38, 38, 36],
		"keysym": ["", "", "", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, -1, 26, -1, -1, 16, -1, -1, 26, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, target="press")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_encode_press_collision():
	raw_key_events_input = pd.DataFrame({
		"time": [0.59, 0.6, 0.66, 0.7, 1.1, 1.21, 1.5],
		"keycode": [27, 38, 27, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "press", "release", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"y": [-1, 26, -1, -1, -1, -1, 16, -1, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, target="press")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_encode_press_consecutive_presses():
	raw_key_events_input = pd.DataFrame({
		"time": [0.62, 0.72, 0.77, 0.9],
		"keycode": [27, 27, 38, 38],
		"keysym": ["", "", "", ""],
		"event": ["press", "release", "press", "release"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, 16, 26, -1, -1, -1, -1, -1, -1, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, target="press")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_encode_press_autopress():
	raw_key_events_input = pd.DataFrame({
		"time": [0.62, 0.72, 0.77, 0.9],
		"keycode": [27, 27, 27, 27],
		"keysym": ["", "", "", ""],
		"event": ["press", "press", "press", "release"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
		"y": [-1, -1, 16, -1, -1, -1, -1, -1, -1, -1, -1]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, target="press")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 52 == result[1]

def test_resample_identification_filter():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 50, 50, 27, 27, 50],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "shift"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"y": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	}, dtype="float32")
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 1 == result[1]

@pytest.mark.skip(reason="currently no multilabel approach in use")
def test_resample_identification_multilabel():
	raw_key_events_input = pd.DataFrame({
		"time": [0.61, 0.7, 0.77, 0.9, 1.1, 1.21, 1.5],
		"keycode": [27, 27, 38, 38, 27, 27, 38],
		"keysym": ["", "", "", "", "", "", ""],
		"event": ["press", "release", "press", "release", "press", "release", "press"]
	})
	raw_key_events_input.set_index("time", inplace=True)
	index_input = pd.Float64Index([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="time")
	encoding_input = "multiclass"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		27: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
		38: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		65: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.encode.resample(raw_key_events_input, index_input, encoding_input, "state")

	pd.testing.assert_frame_equal(expected, result[0])
	assert 53 == result[1]
