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
Avoid copy&paste. Requires prior run of apply models on binary and multiclass models.
"""

import os

import pandas as pd

def full_table(df_mean, df_std):
	# reorder classifiers
	df_mean = df_mean.reindex(["resnet11", "resnet", "CRNN", "WaveNet"])
	df_std = df_std.reindex(["resnet11", "resnet", "CRNN", "WaveNet"])
	print(df_mean.columns.to_list())
	for (r, r_mean), (_, r_std) in zip(df_mean.iterrows(), df_std.iterrows()):
		i = 0
		print(f"{r}:   \t ", end="")
		for (_, mean), (_, std) in zip(r_mean.iteritems(), r_std.iteritems()):
			print(f"{mean*100:.1f} ({std*100:.1f})", end="")
			if i >= len(df_mean.columns)-1:
				break
			if mean < 0.1:
				print(" ", end="")
			print(" & ", end="")
			if std < 0.1:
				print(" ", end="")
			i += 1
		print("\\\\")

def reduced_table(df_mean, df_std):
	# reorder classifiers
	df_mean = df_mean.reindex(["resnet11", "resnet", "CRNN", "WaveNet"], level="classifier")
	df_std = df_std.reindex(["resnet11", "resnet", "CRNN", "WaveNet"], level="classifier")
	try:
		df_mean = df_mean.reindex(["random", "pwgen", "xkcd", "insecure"], level="task_type")
		df_std = df_std.reindex(["random", "pwgen", "xkcd", "insecure"], level="task_type")
	except KeyError:
		df_mean = df_mean.reindex(["touch", "hybrid"], level="typing_style")
		df_std = df_std.reindex(["touch", "hybrid"], level="typing_style")
	df_mean = df_mean.unstack(level="classifier")
	df_std = df_std.unstack(level="classifier")
	# make iterrows work
	df_mean.columns.droplevel()
	df_std.columns.droplevel()
	print(df_mean.columns.to_list())
	for (r, r_mean), (_, r_std) in zip(df_mean.iterrows(), df_std.iterrows()):
		i = 0
		print(f"{r}:   \t ", end="")
		for (_, mean), (_, std) in zip(r_mean.iteritems(), r_std.iteritems()):
			print(f"{mean*100:.1f} ({std*100:.1f})", end="")
			if i >= len(df_mean.columns)-1:
				break
			if mean < 0.1:
				print(" ", end="")
			print(" & ", end="")
			if std < 0.1:
				print(" ", end="")
			i += 1
		print("\\\\")

def main(
		save_path: "path to a directory to save plots at" = "results/analysis",
		save_only: "save the plots and do not show them" = False,
	):

	run = ""
	try:
		df = pd.read_pickle(os.path.join("results/results_bin_default", "binary_0.pkl"))

		# prefilter interesting data
		df = df.loc[df["method"] == "end-to-end"]
		df["classifier"] = df["classifier"].replace({"WaveNet\nadaptation": "WaveNet"})
		df["typing_style"] = df["typing_style"].replace({"touch_typing": "touch"})

		score = "bal_acc"
		# max and min performance of classifiers per participant
		for c in df["classifier"].unique():
			dfc = df.loc[df["classifier"] == c]
			df_mean = dfc.groupby(by=["classifier","participant"])[score].mean()
			df_std = dfc.groupby(by=["classifier","participant"])[score].std()
			print(c)
			print(f"\tparticipant max: {df_mean.max()*100:.1f} ({df_std.iloc[df_mean.argmax()]*100:.1f})")
			print(f"\tparticipant min: {df_mean.min()*100:.1f} ({df_std.iloc[df_mean.argmin()]*100:.1f})")

		score_names = ["bal_acc", "f1", "prec", "rcall"]
		# mean and std performance of classifiers over all data
		df_mean = df.groupby(by=["classifier"])[score_names].mean()
		df_std = df.groupby(by=["classifier"])[score_names].std()
		print("binary all:")
		full_table(df_mean, df_std)
		score_names = ["bal_acc", "f1"]
		df = df.loc[df["classifier"].isin(["resnet11", "CRNN"])]
		# mean and std performance of classifiers per typing style
		df_mean = df.groupby(by=["classifier", "typing_style"])[score_names].mean()
		df_std = df.groupby(by=["classifier", "typing_style"])[score_names].std()
		print("binary typing style:")
		reduced_table(df_mean, df_std)
		# mean and std performance of classifiers per password type
		df_mean = df.groupby(by=["classifier", "task_type"])[score_names].mean()
		df_std = df.groupby(by=["classifier", "task_type"])[score_names].std()
		print("binary password type:")
		reduced_table(df_mean, df_std)
	except FileNotFoundError:
		print("Could not find binary data.")
		pass

	try:
		df_emg = pd.read_pickle(os.path.join("results/results_bin_sensortest/emg", "binary_0.pkl"))
		df_acc = pd.read_pickle(os.path.join("results/results_bin_sensortest/acc", "binary_0.pkl"))
		df_gyro = pd.read_pickle(os.path.join("results/results_bin_sensortest/gyro", "binary_0.pkl"))

		def prefilter(df): # prefilter interesting data
			df = df.loc[df["method"] == "end-to-end"]
			df["classifier"] = df["classifier"].replace({"WaveNet\nadaptation": "WaveNet"})
			df["typing_style"] = df["typing_style"].replace({"touch_typing": "touch"})
			return df

		df_emg = prefilter(df_emg)
		df_acc = prefilter(df_acc)
		df_gyro = prefilter(df_gyro)
		score_names = ["bal_acc", "f1", "prec", "rcall"]

		# mean and std performance of classifiers over all data
		for i, df in enumerate([df_emg, df_acc, df_gyro]):
			df_mean = df.groupby(by=["classifier"])[score_names].mean()
			df_std = df.groupby(by=["classifier"])[score_names].std()
			print(f"binary {'emg' if i == 0 else 'acc' if i == 1 else 'gyro'} all:")
			full_table(df_mean, df_std)
	except FileNotFoundError:
		print("Could not find binary sensor data.")
		pass

	try:
		df = pd.read_pickle(os.path.join("results/results_mc_default", "multiclass_0.pkl"))

		# prefilter interesting data
		df = df.loc[df["method"] == "end-to-end"]
		df = df.loc[df["classifier"] != "CNN (Lab)"]
		df["classifier"] = df["classifier"].replace({"WaveNet\nadaptation": "WaveNet"})
		df["typing_style"] = df["typing_style"].replace({"touch_typing": "touch"})

		score_names = ["acc", "top3_acc", "top5_acc", "top10_acc", "top25_acc"]
		# mean and std performance of classifiers over all data
		df_mean = df.groupby(by=["classifier"])[score_names].mean()
		df_std = df.groupby(by=["classifier"])[score_names].std()
		print("multi-class all:")
		full_table(df_mean, df_std)
		df = df.loc[df["classifier"].isin(["resnet11", "CRNN"])]
		# mean and std performance of classifiers per typing style
		df_mean = df.groupby(by=["classifier", "typing_style"])[score_names].mean()
		df_std = df.groupby(by=["classifier", "typing_style"])[score_names].std()
		print("multi-class typing style:")
		reduced_table(df_mean, df_std)
		# mean and std performance of classifiers per password type
		df_mean = df.groupby(by=["classifier", "task_type"])[score_names].mean()
		df_std = df.groupby(by=["classifier", "task_type"])[score_names].std()
		print("multi-class password type:")
		reduced_table(df_mean, df_std)
	except FileNotFoundError:
		print("Could not find multiclass data.")
		pass

	try:
		df_emg = pd.read_pickle(os.path.join("results/results_mc_sensortest/emg", "multiclass_0.pkl"))
		df_acc = pd.read_pickle(os.path.join("results/results_mc_sensortest/acc", "multiclass_0.pkl"))
		df_gyro = pd.read_pickle(os.path.join("results/results_mc_sensortest/gyro", "multiclass_0.pkl"))
		df_emggyro = pd.read_pickle(os.path.join("results/results_mc_sensortest/emggyro", "multiclass_0.pkl"))

		def prefilter(df): # prefilter interesting data
			df = df.loc[df["method"] == "end-to-end"]
			df = df.loc[df["classifier"] != "CNN (Lab)"]
			df["classifier"] = df["classifier"].replace({"WaveNet\nadaptation": "WaveNet"})
			df["typing_style"] = df["typing_style"].replace({"touch_typing": "touch"})
			return df

		df_emg = prefilter(df_emg)
		df_acc = prefilter(df_acc)
		df_gyro = prefilter(df_gyro)
		df_emggyro = prefilter(df_emggyro)
		score_names = ["acc", "top3_acc", "top5_acc", "top10_acc", "top25_acc"]

		# mean and std performance of classifiers over all data
		for i, df in enumerate([df_emg, df_acc, df_gyro, df_emggyro]):
			df_mean = df.groupby(by=["classifier"])[score_names].mean()
			df_std = df.groupby(by=["classifier"])[score_names].std()
			print(f"multi-class {'emg' if i == 0 else 'acc' if i == 1 else 'gyro' if i == 2 else 'emggyro'} all:")
			full_table(df_mean, df_std)
	except FileNotFoundError:
		print("Could not find multiclass sensor data.")
		pass
