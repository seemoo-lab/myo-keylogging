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
		path: "the path to the parent directory in which to find the pickled DataFrame" = "results/results_bin_default",
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
	):

	run = ""
	try:
		df = pd.read_pickle(os.path.join(path, "binary_0.pkl"))
		run = "binary"
	except FileNotFoundError:
		pass
	try:
		df = pd.read_pickle(os.path.join(path, "multiclass_0.pkl"))
		run = "multiclass"
	except FileNotFoundError:
		pass

	df = df.loc[df["method"] == "end-to-end"]
	df = df.loc[df["classifier"] != "CNN (Lab)"]

	# plot end-to-end result for all participants in individual plots
	for c in classifiers:
		x = "participant"
		hue = "typing_style"
		col = "classifier"
		title=f"model performance end-to-end per user ({run}, {c})"
		filename = f"model_performance_per_user_{run}_end-to-end_{c}"
		data = df.loc[df["classifier"] == c].copy()
		if run == "binary":
			metrics = ["bal_acc"]
			for metric in metrics:
				y = f"model performance ({metric})"
				utils.plotf(utils.plot_metric_results_bars_with_dodge,
						data.rename(columns={metric: y}), x, y, hue=hue,
						title=title, path=save_path,
						filename=f"{filename}_{metric}" if save_only else "")
		if run == "multiclass":
			top_n_metrics = ["top3_acc", "top5_acc", "top10_acc", "top25_acc"]
			y = "model performance"
			utils.plotf(utils.stacked_bars,
					data, x, y, stacks=top_n_metrics, hue=hue,
					title=title, path=save_path,
					filename=f"{filename}_top_n" if save_only else "")
