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
Analyse passwords entered during the second part of the data study.
"""

import glob
import json

import pandas as pd

def eval_diff(df):
	for i in range(8):
		task_df = df.loc[df["task"] == i]
		de_pws = task_df.loc[task_df["layout"] == "de"]
		us_pws = task_df.loc[task_df["layout"] == "us"]
		if task_df["pw1"].nunique() == 1 and task_df["pw2"].nunique() == 1:
			print(f"Task {i} is ok")
		elif (de_pws["pw1"].nunique() == 1 and de_pws["pw2"].nunique() == 1
				and us_pws["pw1"].nunique() == 1 and us_pws["pw2"].nunique() == 1):
			print(f"Task {i} is separated into de and us both being ok")
		elif (task_df["pw1"].unique()[0] in task_df["pw2"].unique()
				and task_df["pw1"].unique()[1] in task_df["pw2"].unique()
				and task_df["pw1"].nunique() == 2 and task_df["pw2"].nunique() == 2):
			print(f"Task {i} is swapped")
		elif (de_pws["pw1"].nunique() == 1 and de_pws["pw2"].nunique() == 1
				and us_pws["pw1"].unique()[0] in us_pws["pw2"].unique()
				and us_pws["pw1"].unique()[1] in us_pws["pw2"].unique()
				and us_pws["pw1"].nunique() == 2 and us_pws["pw2"].nunique() == 2):
			print(f"Task {i} is separated into de and us and us is swapped")
		elif (de_pws["pw1"].unique()[0] in de_pws["pw2"].unique()
				and de_pws["pw1"].unique()[1] in de_pws["pw2"].unique()
				and de_pws["pw1"].nunique() == 2 and de_pws["pw2"].nunique() == 2
				and us_pws["pw1"].nunique() == 1 and us_pws["pw2"].nunique() == 1):
			print(f"Task {i} is separated into de and us and de is swapped")
		elif (de_pws["pw1"].unique()[0] in de_pws["pw2"].unique()
				and de_pws["pw1"].unique()[1] in de_pws["pw2"].unique()
				and de_pws["pw1"].nunique() == 2 and de_pws["pw2"].nunique() == 2
				and us_pws["pw1"].unique()[0] in us_pws["pw2"].unique()
				and us_pws["pw1"].unique()[1] in us_pws["pw2"].unique()
				and us_pws["pw1"].nunique() == 2 and us_pws["pw2"].nunique() == 2):
			print(f"Task {i} is separated into de and us and both are swapped")
		else:
			print(task_df)

def main(path: "the path to the password recordings" = "test-data/"):

	# get prefix of all participants
	pfiles = glob.glob(f"{path}*t0.meta.json")
	print(f"{len(pfiles)} participants")
	pfiles.sort()

	df = pd.DataFrame(columns=["pw1", "pw2", "task", "user", "collection", "layout"])

	for name in pfiles: # for all participants
		name = name[:-12]
		files = glob.glob(f"{name}*.meta.json")
		files.sort()

		for filename in files: # for all tasks
			print(filename)
			with open(filename) as jsonfile:
				json_data = json.load(jsonfile)

			df = df.append({
				"pw1": json_data["task"]["passwords"][0],
				"pw2":  json_data["task"]["passwords"][1],
				"task": int(filename.split(".")[-3][1:]),
				"user": json_data["id"]["user"],
				"collection": json_data["id"]["collection"],
				"layout": json_data["keyboard"]["layout"]
			}, ignore_index=True)

	df = df.sort_values(by=["task", "layout", "user"])

	df = df.loc[df["task"] < 8]
	print(df.drop(columns=["collection"]))

	print("*" * 70)

	df_new = df.drop_duplicates(["pw1", "pw2", "task"])

	print(df_new.drop(columns=["collection"]))

	print("*" * 70)

	eval_diff(df.drop(columns=["collection"]))
