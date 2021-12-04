# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2020-2021  Matthias Gazzari, Annemarie Mattmann
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
Plot the similarity of sensor data of the same and different users in multiple recordings.
"""

import itertools

import pandas as pd
from tslearn.metrics import dtw
from joblib import Memory

import preprocess

MEM = Memory("cache", verbose=0)

def get_word_timings(times, keys, list_keycodes):
	# extract timings for given array of keycodes
	word_timings = []
	word = []
	start_time = 0
	for time, y in zip(times, keys):
		keycode = y
		if keycode == list_keycodes[len(word)]: # matching keycode
			word.append(keycode)
			if len(word) == 1:
				start_time = time
		elif keycode != 0: # different keycode
			# (no releases so a pure typing should have no backspace/shift)
			#if len(word) >= 2:
			#	print(word)
			#	print(keycode)
			word = []
		if word == list_keycodes:
			word_timings.append([start_time-0.005, time+0.01])
			word = []
	#print(f"{len(word_timings)} occurrences found")
	return word_timings

def calc_dtw_of(a, b, dtw_scores, channels):
	dtw_scores["record_1"].append(a.iloc[0]["collection"])
	dtw_scores["record_2"].append(b.iloc[0]["collection"])
	dtw_scores["record_comb"].append(f"{a.iloc[0]['collection']} + {b.iloc[0]['collection']}" if a.iloc[0]['collection'] != b.iloc[0]["collection"] else str(a.iloc[0]['collection']))
	dtw_scores["user_1"].append(a.iloc[0]["user"])
	dtw_scores["user_2"].append(b.iloc[0]["user"])
	dtw_scores["typing style"].append(f"{a.iloc[0]['typing style']} + {b.iloc[0]['typing style']}" if a.iloc[0]['typing style'] != b.iloc[0]['typing style'] else a.iloc[0]['typing style'])
	dtw_scores["same_user"].append("yes" if a.iloc[0]["collection"] != b.iloc[0]["collection"] and a.iloc[0]["user"] == b.iloc[0]["user"] else "no")
	non_data_columns = ["time", "keys", "collection", "user", "typing style"]
	left_channels = [label for label in list(a) if label.endswith("_l")]
	right_channels = [label for label in list(a) if label.endswith("_r")]
	emg_channels = [label for label in list(a) if label.startswith("emg")]
	gyro_channels = [label for label in list(a) if label.startswith("gyro")]
	acc_channels = [label for label in list(a) if label.startswith("acc")]
	unused_channels = [label for label in list(a) if (label.endswith("_l") or label.endswith("_r")) and label[-3] not in channels]
	unused_channels += [label for label in list(a) if label.startswith("quat")]

	temp_0 = a.drop(non_data_columns, axis=1)
	temp_1 = b.drop(non_data_columns, axis=1)
	dtw_scores["dtw_all"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + right_channels + gyro_channels + acc_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + right_channels + gyro_channels + acc_channels + unused_channels, axis=1)
	dtw_scores["dtw_emg_left"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + right_channels + emg_channels + acc_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + right_channels + emg_channels + acc_channels + unused_channels, axis=1)
	dtw_scores["dtw_gyro_left"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + right_channels + emg_channels + gyro_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + right_channels + emg_channels + gyro_channels + unused_channels, axis=1)
	dtw_scores["dtw_acc_left"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + left_channels + gyro_channels + acc_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + left_channels + gyro_channels + acc_channels + unused_channels, axis=1)
	dtw_scores["dtw_emg_right"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + left_channels + emg_channels + acc_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + left_channels + emg_channels + acc_channels + unused_channels, axis=1)
	dtw_scores["dtw_gyro_right"].append(dtw(temp_0, temp_1))

	temp_0 = a.drop(non_data_columns + left_channels + emg_channels + gyro_channels + unused_channels, axis=1)
	temp_1 = b.drop(non_data_columns + left_channels + emg_channels + gyro_channels + unused_channels, axis=1)
	dtw_scores["dtw_acc_right"].append(dtw(temp_0, temp_1))

	return dtw_scores

@MEM.cache
def within_user_dtw(dfs_channelwise, same_user_different_collection, dtw_scores, channels):
	# within user dtws
	for collection in [k for k in dfs_channelwise.keys() if not k in same_user_different_collection]:
		print(f"Comparing record {collection} internally.")
		for subset in itertools.combinations(dfs_channelwise[collection], 2):
			dtw_scores = calc_dtw_of(subset[0], subset[1], dtw_scores, channels)

	# temporary sort those to the end solution...
	for collection in same_user_different_collection:
		print(f"Comparing record {collection} internally.")
		for subset in itertools.combinations(dfs_channelwise[collection], 2):
			dtw_scores = calc_dtw_of(subset[0], subset[1], dtw_scores, channels)

	pd.DataFrame.hist(pd.DataFrame(dtw_scores).drop(["record_1", "record_2", "user_1", "user_2", "same_user", "typing style", "dtw_all"], axis=1), by="record_comb")

	# within user but from different recordings dtw
	for subset in itertools.combinations(dfs_channelwise[same_user_different_collection[0]] + dfs_channelwise[same_user_different_collection[1]], 2):
		if subset[0].iloc[0]['collection'] != subset[1].iloc[0]['collection']:
			print(f"Comparing record {round(subset[0].iloc[0]['collection'])} with {round(subset[1].iloc[0]['collection'])}.")
			dtw_scores = calc_dtw_of(subset[0], subset[1], dtw_scores, channels)

	df_dtw = pd.DataFrame(dtw_scores)
	print(df_dtw)
	#df_dtw = df_dtw.sort_values(by=["record_1", "record_2"], ignore_index=True)
	#print(df_dtw)
	return df_dtw, dtw_scores

@MEM.cache
def between_user_dtw(dfs_channelwise, same_user_different_collection, dtw_scores, channels):
	for collection in [k for k in dfs_channelwise.keys() if not k in same_user_different_collection]:
		for i in same_user_different_collection:
			print(f"Comparing record {i} with {collection}.")
			for same in dfs_channelwise[i]:
				for other in dfs_channelwise[collection]:
					dtw_scores = calc_dtw_of(same, other, dtw_scores, channels)

	df_dtw_between = pd.DataFrame(dtw_scores)
	with pd.option_context("display.max_rows", None, "display.max_columns", None, "expand_frame_repr", False, "display.float_format", lambda x: "%.1f" % x):
		print(df_dtw_between)
	#print(df_dtw_between.loc[df_dtw_between["record_comb"].str.contains("+")])
	return df_dtw_between, dtw_scores

def normalize_df(df):
	# normalize output
	for dtw_col in [col for col in df.columns if col.startswith("dtw")]:
		df[dtw_col] /= df[dtw_col].max()
	return df

def reorganize_data(data_list, list_keycodes, show_plot):
	dfs = []
	dfs_channelwise = {}
	for fused in data_list:
		# reorganize df for plotting
		compare_collections = [16, 29, 3]
		if show_plot and fused["collection"][0] in compare_collections:
			# add hand column to dfs
			left_channels = [label for label in list(fused) if label.endswith("_l")]
			right_channels = [label for label in list(fused) if label.endswith("_r")]
			left_df = fused.drop(right_channels, axis=1)
			left_df = left_df.rename(columns=dict(zip(left_channels, [label.replace("_l", "") for label in left_channels])))
			left_df["hand"] = "left"
			right_df = fused.drop(left_channels, axis=1)
			right_df = right_df.rename(columns=dict(zip(right_channels, [label.replace("_r", "") for label in right_channels])))
			right_df["hand"] = "right"
			# concatenate left and right hand data
			df = left_df.append(right_df, ignore_index=True)
			# stack sensor channels
			df = df.melt(id_vars=["time", "hand", "keys", "collection", "user", "typing style"], var_name="channel", value_name="value")
			#print(df)

		word_timings = get_word_timings(fused["time"], fused["keys"], list_keycodes)

		# cut words from dfs
		for timings in word_timings:
			if show_plot and fused["collection"][0] in compare_collections:
				dfs.append(df.loc[(df["time"] >= timings[0]) & (df["time"] <= timings[1])].copy())
			dfs_channelwise.setdefault(fused["collection"][0], []).append(fused.loc[(fused["time"] >= timings[0]) & (fused["time"] <= timings[1])].copy())
	return dfs, dfs_channelwise

def test_filter(prefix):
	meta_data = preprocess.utils.read_meta(prefix)
	collection = meta_data["id"]["collection"]
	return (collection == 16 and ".t10" in prefix) or (collection == 29 and ".t6" in prefix)

def quetzalcoatl_filter(prefix):
	meta_data = preprocess.utils.read_meta(prefix)
	if meta_data["keyboard"]["layout"] == "us":
		return False
	task_type = meta_data["common"]["task_type"]
	#return task_type == "text"
	return ".t10" in prefix or ".t6" in prefix

def load_data(file_filter, path="train-data/"):
	X, y, X_labels, _, meta_datas = preprocess.all_tasks_np(path, file_filter, method="nearest", cut=False, encoding="multiclass", target="press")
	# transform data back into a list of panda dataframes, enhance with meta_data and fake time
	data_list = []
	for sensor, key, meta_data in zip(X, y, meta_datas):
		df = pd.DataFrame(sensor[0:], columns=X_labels)
		df["time"] = df.index / 200 # roughly 200 Hertz sampling rate
		df["keys"] = key
		df["keys"] = df["keys"].apply(lambda x: preprocess.encode.MULTICLASS_TO_KEYCODE[x])
		df["collection"] = meta_data["id"]["collection"]
		df["user"] = meta_data["id"]["user"]
		df["typing style"] = meta_data["common"]["typing_style"]
		data_list.append(df)
	return data_list

def main(
		path: "path to the training data directory" = "train-data/",
		list_keycodes: "the word's keycodes" = [24, 30, 26, 28, 29, 38, 46, 54, 32, 38, 28, 46], # Quetzalcoatl
		sensor_types: "the sensor types to plot" = ["emg", "gyro", "acc"],#, "imu"],
		channels: "the data channel(s) to plot" = ("0","1","2","3","4","5","6","7"),
		show_plot: "plot result" = False,
		save_only: "save the plots and do not show them" = False
	):
	channels = tuple(channels)
	#ag -A 12 "q,press" record-{16,29}.t{2,6,10,14,18,22}.key.csv
	#seed 1: record-16.t10.key.csv (5x Quetzalcoatl, 1x Quetzalvogels), record-16.t22.key.csv (2x Quetzal)
	#seed 0: record-29.t6.key.csv (5x Quetzalcoatl, 1x Quetzalvogels), record-29.t18.key.csv (2x Quetzal)
	# Quetzalvogels would not be recognized, just in case Quetzal is of interest any time in the future...

	#data_list = load_data(test_filter) # only load a subset
	data_list = load_data(quetzalcoatl_filter) # load all data with Quetzalcoatl
	dfs, dfs_channelwise = reorganize_data(data_list, list_keycodes, show_plot)

	# gyro / 2000
	# (emg + 0.5) / 127.5
	# (acc + 0.5) / 15.5

	# dynamic time warping
	dtw_scores = {"record_1": [], "record_2": [], "record_comb": [], "user_1": [], "user_2": [], "same_user": [], "typing style": [], "dtw_emg_left": [], "dtw_gyro_left": [], "dtw_acc_left": [], "dtw_emg_right": [], "dtw_gyro_right": [], "dtw_acc_right": [], "dtw_all": []}
	same_user_different_collection = (16, 29)

	df_dtw, dtw_scores = within_user_dtw(dfs_channelwise, same_user_different_collection, dtw_scores, channels)

	df_dtw_between, dtw_scores = between_user_dtw(dfs_channelwise, same_user_different_collection, dtw_scores, channels)

	#with pd.option_context("display.max_rows", None, "display.max_columns", None, "expand_frame_repr", False, "display.float_format", lambda x: "%.1f" % x):
	#	print(df_dtw_between)

	# color highlighting based on observed similarity between data - hard-coded!
	df_dtw_between["color_sampler_16"] = df_dtw_between["record_comb"]
	choices = ["16", "16 + 29", "16 + 22", "16 + 0"] # 12 #["16", "16 + 29", "16 + 22", "16 + 5", "16 + 11"] # + 7 and 8
	df_dtw_between.loc[~df_dtw_between["color_sampler_16"].isin(choices), "color_sampler_16"] = "other"
	df_dtw_between["color_sampler_29"] = df_dtw_between["record_comb"]
	choices = ["29", "16 + 29", "29 + 22", "29 + 0"] #
	df_dtw_between.loc[~df_dtw_between["color_sampler_29"].isin(choices), "color_sampler_29"] = "other"

	# plot dtw between collections samples data
	x = "record_comb"
	hue = "same_user"
	y = "dtw_emg_left"
	df_dtw_between_with_same = normalize_df(df_dtw_between.loc[(df_dtw_between["record_1"] != df_dtw_between["record_2"]) | ((df_dtw_between["record_1"] == df_dtw_between["record_2"]) & (df_dtw_between["record_1"].isin(same_user_different_collection)))].sort_values(by=x))

	from . import utils
	hue = "color_sampler_29"
	data = normalize_df(df_dtw_between_with_same.loc[(df_dtw_between_with_same["record_comb"].str.startswith("29")) | (df_dtw_between_with_same["record_comb"].str.endswith("29"))])
	data = data.sort_values(by=["color_sampler_29"], ascending=False) # sort to have others at the top
	data["color_sampler_29"] = data["color_sampler_29"].replace({"16 + 29": "29 + 16"})
	data["color_sampler_29"] = data["color_sampler_29"].replace({"29 + 16": "same person"})
	data["color_sampler_29"] = data["color_sampler_29"].replace({"29": "same recording"})
	data["color_sampler_29"] = data["color_sampler_29"].replace({"29 + 22": "different person (a)"})
	data["color_sampler_29"] = data["color_sampler_29"].replace({"29 + 0": "different person (b)"})
	x = "DTW distance EMG right"
	y = "DTW distance EMG left"
	data = data.rename(columns={"dtw_emg_right": x})
	data = data.rename(columns={"dtw_emg_left": y})
	utils.plotf(utils.show_dtw_users, data, x=x, y=y, hue=hue, style=hue,
				title="between user dtw",
				path="results/analysis", filename="dtw_emg" if save_only else "")
	utils.plt.figure()
	utils.sns.boxplot(data=data, x="color_sampler_29", y=y, hue=hue)
	x = "DTW distance acc right"
	y = "DTW distance acc left"
	data = data.rename(columns={"dtw_acc_right": x})
	data = data.rename(columns={"dtw_acc_left": y})
	utils.plotf(utils.show_dtw_users, data, x=x, y=y, hue=hue, title="between user dtw", style=hue,
				path="results/analysis", filename="dtw_acc" if save_only else "")
	utils.plt.figure()
	utils.sns.boxplot(data=data, x="color_sampler_29", y=y, hue=hue)
	x = "DTW distance gyro right"
	y = "DTW distance gyro left"
	data = data.rename(columns={"dtw_gyro_right": x})
	data = data.rename(columns={"dtw_gyro_left": y})
	utils.plotf(utils.show_dtw_users, data, x=x, y=y, hue=hue, title="between user dtw", style=hue,
				path="results/analysis", filename="dtw_gyro" if save_only else "")
	utils.plt.figure()
	utils.sns.boxplot(data=data, x="color_sampler_29", y=y, hue=hue)

	if save_only:
		return
	utils.plt.show()
