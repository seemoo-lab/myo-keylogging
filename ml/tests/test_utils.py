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

import utils

import numpy as np

def test_top_n_accuracy_single_entry():
	y_score_input = np.array([0, 0.5, 0.1, 0.13, 0.2, 0.03, 0.04, 0])
	y_test_input = np.array(3)
	ns_input = [1,3,5,10,25]

	expected = [0,1,1,1,1]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)

def test_top_n_accuracy_single_entry_top1():
	y_score_input = np.array([0, 0.5, 0, 0.3, 0, 0.2, 0, 0])
	y_test_input = np.array(1)
	ns_input = [1,3,5,10,25]

	expected = [1,1,1,1,1]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)

def test_top_n_accuracy_single_entry_top_last():
	y_score_input = np.array([0, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2, 0.1])
	y_test_input = np.array(0)
	ns_input = [1,3,5,10]

	expected = [0,0,0,1]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)

def test_top_n_accuracy_single_entry_topn_greater_given():
	y_score_input = np.array([0, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2, 0.1])
	y_test_input = np.array(0)
	ns_input = [1,3,5]

	expected = [0,0,0]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)

def test_top_n_accuracy_single_entry_equal_proba_last():
	y_score_input = np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5])
	y_test_input = np.array(7)
	ns_input = [1,3,5,10,25]

	expected = [1,1,1,1,1]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)

def test_top_n_accuracy_single_entry_equal_proba_first():
	y_score_input = np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5])
	y_test_input = np.array(1)
	ns_input = [1,3,5,10,25]

	expected = [0,1,1,1,1]
	result = utils.top_n_accuracy_single_entry(y_score_input, y_test_input, ns_input)

	assert(expected == result)



def test_top_n_accuracy():
	y_score_input = np.array([[0, 0.5, 0.2, 0.3],[0.9, 0.1, 0, 0]])
	y_test_input = np.array([2, 0])
	ns_input = [1,3,5]

	expected = [0.5,1,1]
	result = utils.top_n_accuracy(y_score_input, y_test_input, ns_input)

	assert(expected == result.tolist())
