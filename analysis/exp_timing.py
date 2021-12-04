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
Plot inter-key-interval and keystroke interval timings.
"""

import pathlib
import json

import pandas as pd
import numpy as np

from . import utils

def analyse_key_events(files, save_only, no_plot, source):
	"""Analyse the inter-key-interval and keystroke interval timings."""
	# prepare dictionaries
	data = {
		"recording": [],
		"style": [],
		"task type": [],
		"key_data_idx": [],
		"key": [],
		"press": [],
		"release": [],
	}
	count = 0

	key_index = 1 # set to 2 for key codes and to 1 for key symbols

	for filename in files:
		# gather meta data
		with open(f"{str(filename)[:-7]}meta.json") as jsonfile:
			json_data = json.load(jsonfile)

		# store values for one file
		recording = json_data["id"]["collection"]
		try:
			task_type = json_data["common"]["task_type"]
		except KeyError:
			task_type = "passwords"
		typing_style = " ".join(json_data["common"]["typing_style"].split("_")).capitalize()
		if not typing_style == "Touch typing":
			typing_style = "Non-touch typing"
		key_data_idx = count # indices of key_data at which each new file starts

		# load key events
		key_events = pd.read_csv(filename, engine="c")

		# merge key presses and releases of the same keystroke
		key_dict = {}
		for event in key_events.itertuples(False, None):
			if event[3] == "release" and event[key_index] in key_dict:
				# append all values
				data["recording"].append(recording)
				data["style"].append(typing_style)
				data["task type"].append(task_type)
				data["key_data_idx"].append(key_data_idx)
				data["key"].append(event[key_index])
				data["press"].append(key_dict.pop(event[key_index]))
				data["release"].append(event[0])
				count += 1
			elif event[3] == "press" and not event[key_index] in key_dict: # ignore auto-press
				key_dict[event[key_index]] = event[0]

	df_data = pd.DataFrame(data)

	# calculate keystroke duration and rp and pp interval
	df_data["duration"] = df_data["release"] - df_data["press"]
	df_data["last release"] = df_data["release"].shift(1)
	df_data["interval"] = abs(df_data["press"] - df_data["last release"])
	df_data["last press"] = df_data["press"].shift(1)
	df_data["press interval"] = abs(df_data["press"] - df_data["last press"])
	# set all intervals between files to NaN
	for i in data["key_data_idx"]:
		df_data.at[i, "interval"] = np.nan

	# there should be no negative intervals with the current method of calculation
	assert df_data.loc[df_data['interval'] < 0].shape[0] == 0
	assert df_data.loc[df_data['press interval'] < 0].shape[0] == 0

	x1 = "duration [s]"
	x2 = "press interval [s]"
	df = df_data.rename(columns={"duration": x1, "press interval": x2})

	# pp-intervals
	hue = "task type"
	df_types = df.sort_values(by=[hue], ascending=False)
	df_types = df_types.replace({"uniform \d": "random", "uniform disappearing \d": "random (memorized)"}, regex=True)
	#utils.plotf(utils.plot_pp_interval,
	#			df_types.dropna(), x=x2, hue=hue, style=hue,
	#			title="pp interval per task type", path="results/analysis", drawline=0.025,
	#			filename=f"pp_intervals_tasks_{source}" if save_only else "")
	utils.plotf(utils.plot_pp_interval,
				df_types.dropna(), x=x2, hue=hue, style=hue, xlim=(-0.02,0.42),
				loc="upper left", drawline=0.025,
				title="pp interval per task type", path="results/analysis",
				filename=f"pp_intervals_tasks_{source}_zoom" if save_only else "")
	# duration
	utils.plotf(utils.plot_pp_interval,
				df_types.dropna(), x=x1, hue=hue, style=hue, xlim=(-0.01,0.21), loc="upper left",
				title="pp durations per task type", path="results/analysis",
				filename=f"pp_durations_tasks_{source}_zoom" if save_only else "")

	if not save_only:
		utils.plt.show()


def main(
		path: "path to the directory to load data from" = "train-data/",
		save_only: "save the plots and do not show them" = False,
		no_plot: "only print, do not plot" = False
	):
	# get all key files
	files = list(pathlib.Path(path).glob("*.key.csv"))
	source = pathlib.Path(path).name
	print(f"Analysing {len(files)} file(s)...")
	analyse_key_events(files, save_only, no_plot, source)
