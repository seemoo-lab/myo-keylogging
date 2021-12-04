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
Print inter-key-interval and keystroke interval timings.
"""

import pandas as pd
import numpy as np
import pathlib
import json

def analyse_key_events(files, source):
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

	df_rec = df_data.groupby(["recording"])["duration", "interval"].agg(np.mean).reset_index()
	print(f"Minimum mean interval (across recordings): {df_rec['interval'].min()}")
	print(f"Maximum mean interval (across recordings): {df_rec['interval'].max()}")
	print(f"Minimum mean duration (across recordings): {df_rec['duration'].min()}")
	print(f"Maximum mean duration (across recordings): {df_rec['duration'].max()}")
	print()
	df_rec = df_data.groupby(["recording"])["duration", "interval"].agg(np.median).reset_index()
	print(f"Minimum median interval (across recordings): {df_rec['interval'].min()}")
	print(f"Maximum median interval (across recordings): {df_rec['interval'].max()}")
	print(f"Minimum median duration (across recordings): {df_rec['duration'].min()}")
	print(f"Maximum median duration (across recordings): {df_rec['duration'].max()}")
	print()
	df_no_auto = df_data.loc[(df_data["duration"] <= 0.5) & (~df_data["key"].isin(["Shift_R", "Shift_L"]))]
	df_rec = df_no_auto.groupby(["recording"])["duration", "interval"].agg(np.mean).reset_index()
	print(f"Minimum mean duration (across recordings, excl. auto-press): {df_rec['duration'].min()}")
	print(f"Maximum mean duration (across recordings, excl. auto-press): {df_rec['duration'].max()}")
	df_rec = df_no_auto.groupby(["recording"])["duration", "interval"].agg(np.median).reset_index()
	print(f"Minimum median duration (across recordings, excl. auto-press): {df_rec['duration'].min()}")
	print(f"Maximum median duration (across recordings, excl. auto-press): {df_rec['duration'].max()}")
	print()
	print(f"Minimal interval (including overlaps): {np.min(abs(df_data['interval'].dropna()))}")
	print(f"Maximum interval (including overlaps): {np.max(abs(df_data['interval'].dropna()))}")
	print(f"Mean interval (including overlaps): {np.mean(abs(df_data['interval'].dropna()))}")
	print(f"Median interval (including overlaps): {np.median(abs(df_data['interval'].dropna()))}")
	print()
	print(f"Minimal duration (including auto-press): {np.min(df_data['duration'])}")
	print(f"Maximum duration (including auto-press): {np.max(df_data['duration'])}")
	print(f"Mean duration (including auto-press): {np.mean(df_data['duration'])}")
	print(f"Median duration (including auto-press): {np.median(df_data['duration'])}")
	print()
	print(f"Minimal duration (excluding auto-press): {np.min(df_no_auto['duration'])}")
	print(f"Maximum duration (excluding auto-press): {np.max(df_no_auto['duration'])}")
	print(f"Mean duration (excluding auto-press): {np.mean(df_no_auto['duration'])}")
	print(f"Median duration (excluding auto-press): {np.median(df_no_auto['duration'])}")
	print()
	def stats_below_limit(limit, freq, value):
		percentage = (df_data.loc[df_data[value] <= limit].shape[0]/df_data.shape[0])*100
		total = df_data.loc[df_data[value] <= limit].shape[0]
		#print(df_data.loc[df_data[value] <= limit]["recording"].value_counts())
		print(df_data.loc[df_data[value] <= limit]["style"].value_counts())
		print(df_data.loc[df_data[value] <= limit]["task type"].value_counts())
		return f"Percentage of {value}s below {limit*1000:.1f} ms (i.e. < {(freq*limit):.1f} samples at {freq} Hz): {percentage:.2f} % (total: {total})"
	print(stats_below_limit(0.0025, 400, "interval"))
	print(stats_below_limit(0.005, 200, "interval"))
	print(stats_below_limit(0.01, 200, "interval"))
	print(stats_below_limit(0.015, 200, "interval"))
	print(stats_below_limit(0.02, 200, "interval"))
	print(stats_below_limit(0.03, 200, "interval"))
	print(stats_below_limit(0.035, 200, "interval"))
	print(stats_below_limit(0.04, 200, "interval"))
	print(stats_below_limit(0.045, 200, "interval"))
	print(stats_below_limit(0.05, 200, "interval"))
	print(stats_below_limit(0.06, 200, "interval"))
	print(stats_below_limit(0.09, 200, "interval"))
	#print(stats_below_limit(0.01, 100, "interval"))
	print()
	print(stats_below_limit(0.01, 200, "duration"))
	print(stats_below_limit(0.015, 200, "duration"))
	print(stats_below_limit(0.02, 200, "duration"))
	print(stats_below_limit(0.025, 200, "duration"))
	print(stats_below_limit(0.03, 200, "duration"))
	print(stats_below_limit(0.035, 200, "duration"))
	print(stats_below_limit(0.04, 200, "duration"))
	print(stats_below_limit(0.045, 200, "duration"))
	print(stats_below_limit(0.05, 200, "duration"))
	print(stats_below_limit(0.06, 200, "duration"))
	print(stats_below_limit(0.09, 200, "duration"))
	print(stats_below_limit(0.1, 200, "duration"))
	print(stats_below_limit(0.135, 200, "duration"))
	print(stats_below_limit(0.15, 200, "duration"))
	print(stats_below_limit(0.16, 200, "duration"))
	print(stats_below_limit(0.2, 200, "duration"))
	#print(stats_below_limit(0.01, 100, "duration"))


def main(
		path: "path to the directory to load data from" = "train-data/"
	):
	# get all key files
	files = list(pathlib.Path(path).glob("*.key.csv"))
	source = pathlib.Path(path).name
	print(f"Analysing {len(files)} file(s)...")
	analyse_key_events(files, source)
