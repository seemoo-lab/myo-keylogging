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
Print/store the class imbalance for a run with the given parameters.
"""

import csv
import pathlib

import numpy as np

import preprocess

def store_data_imbalance(path, csv_file, user, task_types, encoding, target, preserve_all, store, cut_to):
	if encoding not in ("binary", "shift"):
		print("encoding needs to be either binary or shift, given {}".format(encoding))
		return
	cut = not preserve_all

	# load and preprocess the data
	_, y, _, _, _ = preprocess.all_tasks_np(
		path=path,
		file_filter=preprocess.utils.FileFilter(user, task_types),
		cut=cut,
		encoding=encoding,
		target=target,
		ignore=["emg","quat","acc","gyro"],
		select=cut_to
	)

	# count uniques
	pos = 0
	neg = 0
	step_size = 1
	print(f"Step size: {step_size}")
	for y_file in y:
		unique, counts = np.unique(y_file[::step_size], return_counts=True)
		if unique[0] == 1 and unique[1] == 0:
			pos += counts[0]
			neg += counts[1]
		elif unique[1] == 1 and unique[0] == 0:
			pos += counts[1]
			neg += counts[0]
		else:
			print("Something somewhere went terribly wrong.")
			return

	if store:
		user = "all" if user == None else user
		task_types = "all" if task_types == [] else task_types[0] if len(task_types) == 1 else task_types
		row = [user, task_types, encoding, target, cut, cut_to, neg, pos, neg/(pos+neg), pos/(pos+neg)]
		file_path = pathlib.Path(csv_file)
		if file_path.is_file():
			# only append the row if the file exists
			with open(csv_file, "a") as csvfile:
				csv_data = csv.writer(csvfile, delimiter=",")
				csv_data.writerow(row)
			print(f"Written to {csv_file}.")
		else:
			# add a header if the file does not exist
			pathlib.Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
			with open(csv_file, "w") as csvfile:
				csv_data = csv.writer(csvfile, delimiter=",")
				csv_data.writerow(["user", "task_types", "encoding", "target", "cut", "split", "neg_samples", "pos_samples", "relative_neg_samples", "relative_pos_samples"])
				csv_data.writerow(row)
			print(f"Created {csv_file}.")
	else:
		print("\nTotal number of positive samples: {}, {} of all data.".format(pos, pos/(pos+neg)))
		print("Total number of negative samples: {}, {} of all data.".format(neg, neg/(pos+neg)))
		if neg > pos:
			print("This is about {} times more negative than positive samples.".format(neg/pos))

def main(
	path: "path to a directory to load data from" = "train-data/",
	csv_file: "path to and name of the csv file to store data in" = "results/train-data-skew.csv",
	user: "the user(s) or typing style to train on (e.g. 1: user with id 1, touch_typing: all touch typists, min: the min sample, default: None: all users)" = None,
	task_types: "the task type(s) to train on (e.g. uniform, default: []: all task types)" = [],
	encoding: "the classification to use (binary, shift, finger or multiclass)" = "binary",
	target: "the target for encoding (state, press or release)" = "press",
	preserve_all: "if False, cut the data to the task, if True preserve all data" = False,
	store: "if True, store the results" = False,
	cut: "cut the data to passwords (first, last, all or None)" = None
):
	print(f"Storing to {csv_file}")
	store_data_imbalance(path, csv_file, user, task_types, encoding, target, preserve_all, store, cut)

