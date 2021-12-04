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
This is a special load result code for apply models to analyse typing speed effects.
"""

import os

import pandas as pd

from . import utils

def main(
		classifiers: "list of classifiers to plot for" = ["CRNN", "resnet11", "resnet", "Wavenet adaptation"],
		path_bin: "" = "results/results_bin_default",
		path_mul: "" = "results/results_mc_default",
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
	):

	run = ""
	try:
		df_bin = pd.read_pickle(os.path.join(path_bin, "binary_0.pkl"))
	except FileNotFoundError:
		print("Could not find binary data.")
		pass
	try:
		df_mul = pd.read_pickle(os.path.join(path_mul, "multiclass_0.pkl"))
	except FileNotFoundError:
		print("Could not find multiclass data.")
		pass

	x = "typing_speed"
	df_bin = df_bin.loc[df_bin["method"] == "end-to-end"]
	df_mul = df_mul.loc[df_mul["method"] == "end-to-end"]
	df_bin = df_bin.loc[df_bin["classifier"].isin(classifiers)]
	df_mul = df_mul.loc[df_mul["classifier"].isin(classifiers)]
	df_bin = df_bin.reset_index(drop=True)
	df_mul = df_mul.reset_index(drop=True)
	# replace wrong typing speed values in df_mul with correct ones in df_bin
	# as typing speed data is calculated on binary
	df_mul[x] = df_bin[x]
	for c in df_bin["classifier"].unique():
		x = "typing_speed"
		data = df_bin.loc[df_bin["classifier"] == c]
		hue = ["f1", "bal_acc"]
		y = "model performance"
		title=f"model performance speed end-to-end (binary, {c})"
		filename = f"model_performance_speed_binary_end-to-end_{c}"
		data = data.rename(columns={x: "typing speed [keys/min]"})
		x = "typing speed [keys/min]"
		utils.plotf(utils.plot_speed_performance_scatter, data, x, y, hue=hue, style=hue,
			draw_line=0.5,
			title=title, path=save_path#, filename=f"{filename}"
		)
		y = "bal_acc"
		hue = "task type"
		title=f"model performance speed end-to-end (binary, {c})"
		filename = f"model_performance_speed_binary_end-to-end_{c}_task_type"
		utils.plotf(utils.plot_speed_performance_scatter, data, x, y, hue=hue, style=hue,
			draw_line=0.5,
			title=title, path=save_path, filename=f"{filename}"
		)
	for c in df_mul["classifier"].unique():
		x = "typing_speed"
		data = df_mul.loc[df_mul["classifier"] == c]
		y = "top3_acc"
		hue = "task type"
		title=f"model performance speed end-to-end (multiclass, {c})"
		filename = f"model_performance_speed_multiclass_end-to-end_{c}_task_type_top_3"
		data = data.rename(columns={x: "typing speed [keys/min]"})
		x = "typing speed [keys/min]"
		utils.plotf(utils.plot_speed_performance_scatter, data, x, y, hue=hue, style=hue,
			draw_line=3/52,
			title=title, path=save_path, filename=f"{filename}"
		)
