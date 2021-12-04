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
This is a special load result code for apply models to load predictions.
"""

import pathlib

import pandas as pd

from . import utils

def main(
		path: "the path to the parent directory in which to find the pickled DataFrame" = "results/results_bin_default",
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
		tolerance: "the temporary tolerance distance to use for reporting and plotting" = 10,
	):
	df_files = list(pathlib.Path(path).glob("*.pkl"))
	for f in df_files:
		if "user" in f.stem:
			df = pd.read_pickle(pathlib.Path(path).joinpath(f"{f.stem}.pkl"))
			filename = f"sample_{'_'.join(f.stem.split('_')[6:])}"
			dist = tolerance
			break

	x = "time [s]"
	y = "probability"
	row = "row"
	hue = "hue"

	# replace estimator names
	df = df.replace({"resnet11": "TSC ResNet11", "resnet": "CWT+ResNet18", "WaveNet adaptation": "TSC WaveNet", "WaveNet\nadaptation": "TSC WaveNet"}, regex=True)
#	df = df.rename(columns={"resnet11 (threshold 0.5)": "TSC ResNet11 (threshold 0.5)", "resnet (threshold 0.5)": "CWT+ResNet18 (threshold 0.5)", "WaveNet adaptation (threshold 0.5)": "TSC WaveNet (threshold 0.5)", "WaveNet\nadaptation (threshold 0.5)": "TSC WaveNet (threshold 0.5)"})
	# reorder estimator names
	df[hue] = df[hue].replace({"truth": 0, "TSC ResNet11": 1, "CWT+ResNet18": 2, "CRNN": 3, "TSC WaveNet": 4, "TSC ResNet11 peak": 5, "CWT+ResNet18 peak": 6, "CRNN peak": 7, "TSC WaveNet peak": 8})
	df = df.sort_values(by=[hue, x])
	df[hue] = df[hue].replace({0: "truth", 1: "TSC ResNet11", 2: "CWT+ResNet18", 3: "CRNN", 4: "TSC WaveNet", 5: "TSC ResNet11 peak", 6: "CWT+ResNet18 peak", 7: "CRNN peak", 8: "TSC WaveNet peak"})

	# create temporal tolerance spans
	truth_times = df.loc[(df["hue"] == "truth") & (df["probability"] == 1)][x]
	span_size = dist * 0.005 # from sample size to ms
	spans=[[[ts-span_size, ts+span_size] for ts in truth_times]] * (len([name for name in df["hue"].unique() if "peak" not in name]) - 1)

	utils.plotf(utils.plot_truth_prediction,
		df, x, y, row=row, hue=row, style=hue, draw_line=0.5, spans=spans,
		title=f"predictions vs ground truth (binary)",
		path=save_path, filename=filename if save_only else ""
	)


