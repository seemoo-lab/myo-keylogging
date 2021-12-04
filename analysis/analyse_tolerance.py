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
This is a special load result code for apply models to analyse participant performance differences.
"""

import os

import pandas as pd

from . import utils

def main(
		classifiers: "list of classifiers to plot for" = ["CRNN", "resnet11", "resnet", "Wavenet adaptation"],
		path: "the path to the parent directory in which to find the pickled DataFrame" = "results/results_chain_default",
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
	):

	run = ""
	try:
		df_bin = pd.read_pickle(os.path.join(path, "binary_0.pkl"))
	except FileNotFoundError:
		print("Could not find binary data, trying results/results_bin_default.")
		try:
			df_bin = pd.read_pickle(os.path.join("results/results_bin_default", "binary_0.pkl"))
		except FileNotFoundError:
			print("Could not find binary data")
			pass
	try:
		df_mul = pd.read_pickle(os.path.join(path, "multiclass_1.pkl"))
	except FileNotFoundError:
		print("Could not find multiclass data.")
		pass

	df_bin = df_bin.loc[df_bin["method"] != "end-to-end"]
	df_mul = df_mul.loc[df_mul["method"] != "end-to-end"]
	df_mul = df_mul.loc[df_mul["classifier"] != "CNN (Lab)"]

	x = "tolerance"
	y = "model performance"
	hue = ["acc", "bal_acc", "f1"]
	# plot for temp 0 to temp 10
	data = df_bin.loc[~df_bin["method"].isin(["temp_20", "temp_100"])].replace({"peak": "0", "temp_": ""}, regex=True)
	data = data.rename(columns={"method": x})
	data["tolerance [s]"] = data[x].astype(float) * 0.005
	x = "tolerance [s]"
	# shift colors to keep red as color for bal acc (needs exact length of palette)
	palette = [utils.sns.color_palette(utils.COLOR_PALETTE)[-3]] + utils.sns.color_palette(utils.COLOR_PALETTE, n_colors=len(hue)-1)
	for c in classifiers:
		utils.plotf(utils.plot_temp_metric_line, data.loc[data["classifier"] == c], x, y, hue=hue,
			style=hue, palette=palette,
			title=f"model performance per distance ({c})", path=save_path,
			filename=f"model_performance_per_distance_{c}" if save_only else ""
		)

	x = "method"
	y = "model performance"
	hue = "metric"
	style = "encoding"
	score_names = sorted(list(set(df_mul.columns) - set(["classifier", "participant", "method", "lags", "task_type", "typing_style", "typing_style", "typing_speed"])))
	# prepare data
	df_bin[style] = "binary"
	df_mul[style] = "multiclass"
	df_bin = df_bin.drop(columns=["acc", "prec", "rcall"])
	df_mul = df_mul.drop(columns=["acc", "f1", "bal_acc", "prec", "rcall", "entropy"])
	rest = list(set(df_bin.columns) - set(score_names))
	df_bin = df_bin.melt(id_vars=rest, var_name=hue, value_name=y)
	rest = list(set(df_mul.columns) - set(score_names))
	df_mul = df_mul.melt(id_vars=rest, var_name=hue, value_name=y)
	# concatenate dfs
	data = pd.concat([df_bin, df_mul], ignore_index=True)
	data = data.loc[data[x].str.startswith("temp")]
	data = data.replace({"temp_": ""}, regex=True)
	data = data.rename(columns={"method": "tolerance"})

	for c in classifiers:
		x = "tolerance"
		data_classifier = data.loc[data["classifier"] == c]
		data_biased = data_classifier.loc[(data_classifier[x].str.contains("bias")) | (data_classifier[hue] == "bal_acc") | (data_classifier[hue] == "f1")]
		data_b = data_biased.replace({"bias_": ""}, regex=True)
		data_b["tolerance [s]"] = data_b[x].astype(float) * 0.005
		x = "tolerance [s]"
		utils.plotf(utils.plot_temp_metric_bin_mul_line,
			data_b.loc[~data_b[hue].isin(["bal_acc_mul", "f1_mul"])], x, y, hue, style,
			path=save_path, title=f"performance for binary vs multiclass - biased ({c})",
			filename=f"performance_bin_mul_distance_bias_{c}" if save_only else ""
		)
