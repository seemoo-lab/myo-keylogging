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
Evaluate the result of a shallow hyperparameter optimization run.
"""

import json

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt

def load_result_file(result_file_name):
	# load json data and flatten the hierarchy with json_normalize
	with open(result_file_name) as json_file:
		json_data = json.load(json_file)
	df = pd.json_normalize(json_data["results"]["hpo"])
	# drop unused columns
	df = df.drop(columns=["mean_fit_time", "std_fit_time", "fit_times", "scores"])
	return df

def print_stats(df):
	print("Number of entries:", len(df))
	print("Columns:", df.columns)
	print()
	print("Correlation with the mean of the test scores:")
	print(df.corr()["mean_test_score"].sort_values())
	print()
	print("Correlation with the standard deviation of the test scores:")
	print(df.corr()["std_test_score"].sort_values())
	print()

def select_params(df, file_name):
	with open(file_name) as json_file:
		param_choice = json.load(json_file)
	for param_name, param_list in param_choice.items():
		if param_name.startswith("params.") and param_list:
			df = df[df[param_name].isin(param_list)]
	return df

def plot_correlations(df, y_label, z_label, selection_file_name, corr_type="spearman"):
	df["variation"] = np.where(df[z_label] < df[z_label].median(), "low", "high")
	# restrict parameters
	df = select_params(df, selection_file_name)
	print("Remaining data points:", len(df))

	corr_matrix = df.corr(corr_type)
	plt.figure()
	sns.heatmap(corr_matrix.abs()).set_title(f"{corr_type} correlation")

	corr_factors = corr_matrix[y_label].reindex(corr_matrix[y_label].abs().sort_values(ascending=False).index)
	print(corr_factors)

	# plot violin and strip plots of the paramters against the mean test score
	# also visualize the standard deviation of the test score and order the plots by the correlation coefficient
	for col, corr in corr_factors.items():
		if col.startswith("params") and df[col].nunique() > 1:
			plt.figure()
			sns.violinplot(
				x=col, y=y_label, data=df,
				#hue="variation", split=True,
				inner="quartile",
				scale="count", scale_hue=False, cut=0
			).set_title(f"{corr_type} correlation factor: {corr}")
			sns.stripplot(x=col, y=y_label, hue="variation", size=15, palette="coolwarm", data=df, color="0.3", alpha=0.3)

def main(
		result_file_name: "file name of the .result.json file from a shallow hyperparameter optimization",
		selection_file_name: "file name of a JSON file containing fixed hyperparameters for narrowing down the search for good hyperparameters"
	):
	df = load_result_file(result_file_name)
	print_stats(df)

	# pick top percentage of parameter sets
	df = df.sort_values(by="mean_test_score", ascending=False)

	# plot correlations
	plot_correlations(df, "mean_test_score", "std_test_score", selection_file_name)
	plt.show()
