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

import preprocess.fuse

def test_merge_interpolate_sensors():
	emg_events_input = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"0": [1, 2, 3, 4, 5, 6, 7, 8, 9],
		"1": [1, 2, 3, 4, 5, 6, 7, 8, 9]
	})
	emg_events_input.set_index("time", inplace=True)
	imu_events_input = pd.DataFrame({
		"time": [0.2, 0.5, 0.9, 1.3, 1.5],
		"0": [00, 11, 55, 99, 000],
		"1": [00, 11, 55, 99, 000]
	})
	imu_events_input.set_index("time", inplace=True)
	leftwards_input = True
	method_input = "index"

	expected = pd.DataFrame({
		"time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
		"0_l": [1, 2, 3, 4, 5, 6, 7, 8, 9],
		"1_l": [1, 2, 3, 4, 5, 6, 7, 8, 9],
		"0_r": [11, 22, 33, 44, 55, 66, 77, 88, 99],
		"1_r": [11, 22, 33, 44, 55, 66, 77, 88, 99]
	})
	expected.set_index("time", inplace=True)
	result = preprocess.fuse.merge_interpolate_sensors(emg_events_input, imu_events_input, leftwards_input, method_input)

	pd.testing.assert_frame_equal(expected, result, check_dtype=False)
