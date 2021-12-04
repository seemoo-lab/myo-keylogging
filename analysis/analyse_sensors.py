# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2021  Matthias Gazzari, Annemarie Mattmann
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
This is a special load result code for apply models to compare sensor type results as these are separate analysis runs.
"""

import os

import pandas as pd

from . import utils

def main(
		path: "the path to the parent directory of the directories in which the sensor dataframe pkls are" = "results/results_bin_sensortest",
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
	):

	x = "sensor type"
	run = ""
	dirs = os.listdir(path)
	dfs = {}
	for d in dirs:
		try:
			dfs[d] = pd.read_pickle(os.path.join(path, d, "binary_0.pkl"))
			metric = "bal_acc"
			run = "binary"
		except FileNotFoundError:
			pass
		try:
			dfs[d] = pd.read_pickle(os.path.join(path, d, "multiclass_0.pkl"))
			metric = "top5_acc"
			run = "multiclass"
		except FileNotFoundError:
			pass
		dfs[d][x] = d
	df = pd.concat(dfs.values(), ignore_index=True)

	df = df.loc[df["method"] == "end-to-end"]
	df = df.loc[df["classifier"] != "CNN (Lab)"]
	y = f"model performance ({metric})"
	hue = "classifier"
	df = df.rename(columns={metric: y})
	#df = df.drop(list(set(df.columns) - set([x, y, hue])), axis=1)
	df = df.replace({"acc": "a", "emg": "e", "gyro": "g", "quat": "q"}, regex=True)
	# replace estimator names
	df = df.replace({"resnet11": "TSC ResNet11", "resnet": "CWT+ResNet18", "WaveNet adaptation": "TSC WaveNet", "WaveNet\nadaptation": "TSC WaveNet"})
	# reorder estimator names
	df = df.replace({"truth": 0, "TSC ResNet11": 1, "CWT+ResNet18": 2, "CRNN": 3, "TSC WaveNet": 4})
	df = df.sort_values(hue)
	df[hue] = df[hue].replace({0: "truth", 1: "TSC ResNet11", 2: "CWT+ResNet18", 3: "CRNN", 4: "TSC WaveNet"})

	utils.plotf(utils.plot_sensor_difference, df, x, y, hue=hue,
					title="compare sensor types", path=save_path,
					filename=f"senor_type_comparison_{run}" if save_only else "")
	df = df.replace({"a": "acc", "e": "emg", "g": "gyro"})
	df = df.sort_values(by=x)
	utils.plotf(utils.plot_sensor_difference, df.loc[df[x].isin(["emg", "acc", "gyro"])], x, y, hue=hue,
					title="compare sensor types", path=save_path,
					filename=f"senor_type_comparison_{run}_eag" if save_only else "")
	top_n_metrics = ["top3_acc", "top5_acc", "top10_acc", "top25_acc"]
	y = "model performance"
#	if run == "multiclass":
#		utils.plotf(utils.stacked_bars_hue, df.loc[df[x].isin(["emg", "acc", "gyro"])], x, y, hue=hue,
#					stacks=top_n_metrics,
#					title="compare sensor types", path=save_path,
#					filename=f"senor_type_comparison_{run}_eag_top_n" if save_only else "")


