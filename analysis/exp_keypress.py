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
Plot aggregated IMU and EMG data over keypresses of selected characters.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import preprocess
from preprocess import utils
from preprocess.encode import KEYMAP_DE, MULTICLASS_TO_KEYCODE

def get_key_name(class_number):
	return KEYMAP_DE[MULTICLASS_TO_KEYCODE[class_number]][0]

def calc_norm(df):
	df["acc_norm_l"] = sum(df[f"acc{i}_l"]**2 for i in range(3))**0.5
	df["acc_norm_r"] = sum(df[f"acc{i}_r"]**2 for i in range(3))**0.5
	df["acc_norm_l"] -= df["acc_norm_l"].mean()
	df["acc_norm_r"] -= df["acc_norm_r"].mean()
	df["gyro_norm_l"] = sum(df[f"gyro{i}_l"]**2 for i in range(3))**0.5
	df["gyro_norm_r"] = sum(df[f"gyro{i}_r"]**2 for i in range(3))**0.5
	df["emg_norm_l"] = sum(df[f"emg{i}_l"]**2 for i in range(8))**0.5
	df["emg_norm_r"] = sum(df[f"emg{i}_r"]**2 for i in range(8))**0.5
	# 1, 0, 7 for flexors, rest for extensor
	df["emg_flexors_norm_l"] = sum(df[f"emg{i}_l"]**2 for i in (1, 0, 7))**0.5
	df["emg_extensors_norm_l"] = sum(df[f"emg{i}_l"]**2 for i in (2,3,4,5,6))**0.5
	df["emg_flexors_norm_r"] = sum(df[f"emg{i}_r"]**2 for i in (1, 0, 7))**0.5
	df["emg_extensors_norm_r"] = sum(df[f"emg{i}_r"]**2 for i in (2,3,4,5,6))**0.5
	return df

def slice_concat(df, search_term, left=60, right=60):
	indices = df[df.y == search_term].index.values
	df_list = []
	key_name = KEYMAP_DE[MULTICLASS_TO_KEYCODE[search_term]][0]
	print(f"number of {key_name} keystrokes: {len(indices)}")
	for i, idx in enumerate(indices):
		start = idx - left # (inclusive)
		end = idx + right + 1 # (not inclusive bound)
		segment = df[start:end].copy()
		segment["time"] = range(-left, right + 1)
		segment["time"] *= 5
		df_list.append(segment)
	sliced_df = pd.concat(df_list)
	return sliced_df

def do_all(path, user, target):
	ff = preprocess.utils.FileFilter(user, [])
	df = preprocess.all_tasks_pd(path, ff, ignore=["quat"], target="press", encoding="multiclass")
	df = calc_norm(df)
	sliced_df = slice_concat(df, target)
	sliced_df = sliced_df.filter(regex="norm|time")
	return sliced_df

def main(
	path: "path to a directory to load data from" = "train-data/",
	users = 1,
	keys = (29, 32),
):
	for key in keys:
		sliced_df = do_all(path, users, key)
		for sensor_type in ("acc", "gyro", "emg", "emg_flexors", "emg_extensors"):
			melted_df = pd.melt(sliced_df.filter(regex=f"{sensor_type}_norm|time"), id_vars="time")
			plt.figure()
			ax = sns.lineplot(data=melted_df, x="time", y="value", hue="variable", ci="sd")
			ax.set_title(f"Magnitude of {sensor_type.capitalize()} data on aggregated ({get_key_name(key)} keystrokes)")

	#left_melted = pd.melt(sliced_df.filter(regex="_l|sample"), id_vars="sample")
	#right_melted = pd.melt(sliced_df.filter(regex="_r|sample"), id_vars="sample")

	#sns.lineplot(data=left_melted, x="sample", y="value", hue="variable")
	#plt.figure()
	#sns.lineplot(data=right_melted, x="sample", y="value", hue="variable")
	#sns.lineplot(data=emg_melted, x="sample", y="value", hue="variable")

	#mean_df = sliced_df.groupby("sample").mean()
	#std_df = sliced_df.groupby("sample").std()
	#mean_df.filter(regex="acc").plot()
	#mean_df.filter(regex="gyro").plot()
	#mean_df.filter(regex="emg").plot()
	plt.show()



