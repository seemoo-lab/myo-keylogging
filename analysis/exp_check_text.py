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
Analyse texts entered during the first part of the data study.
"""

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("notebook", rc={"font.size":11, "axes.titlesize":11, "axes.labelsize":11})

from preprocess.encode import KEYMAP_DE, KEYMAP_US

def get_data(layout, filename, ignore_header):
	# load key maps
	key_map = KEYMAP_US if layout == "us" else KEYMAP_DE
	for key in (22, 36, 50, 62, 65):
		key_map.pop(key)
	keys = tuple(key_map.values()) + (" ", "\n")

	# read texts
	texts = []
	with open(filename, "r") as f:
		texts = f.read().splitlines()

	# create an array of text extracts (i.e. paragraphs)
	extracts = []
	start = 0
	for end, text in enumerate(texts):
		if text == "":
			extracts.append(texts[start:end])
			start = end + 1
	rest = texts[start:]
	if rest:
		extracts.append(rest)

	# count key occurences
	key_occurences = [[0 for col in range(len(keys))] for row in range(len(extracts))]
	unknown = [[] for row in range(len(extracts))]
	num_sharp_s = 0
	num_shift = 0
	for t, text in enumerate(extracts):
		for s, sentence in enumerate(text):
			if s == 0 and ignore_header:
				continue
			for character in sentence:
				if character == "ß":
					num_sharp_s += 1
				for i, tuples in enumerate(keys):
					if character in tuples[-1]:
						num_shift += 1
					if character in tuples:
						key_occurences[t][i] += 1
						break
				else:
					unknown[t].append(character)

	total_key_occurences = [sum(map(int, array)) for array in zip(*key_occurences)]
	total_keys = sum(map(int, total_key_occurences))
	letter_indices = [index for index, value in enumerate(get_key_mapping(total_key_occurences, keys)) if value[-2].isalpha()]
	total_letter_occurences = [total_key_occurences[i] for i in letter_indices]
	letters = [keys[i] for i in letter_indices]
	total_letters = sum(map(int, total_letter_occurences))

	return keys, letters, key_occurences, unknown, num_sharp_s, num_shift, total_key_occurences, total_keys, total_letter_occurences, total_letters

def get_key_mapping(array, keys):
	return [f"{value} of [{keys[t][0]}, {keys[t][-1]}]" for t, value in enumerate(array)]

def get_below_minimum_key_mapping(array, keys, min_occurrence):
	return [f"{value} of [{keys[t][0]}, {keys[t][-1]}]" for t, value in enumerate(array) if value < min_occurrence]

def print_data(keys, letters, key_occurences, unknown, num_sharp_s, num_shift, total_key_occurences, total_keys, total_letter_occurences, total_letters, min_occurrence):
	for t, array in enumerate(key_occurences):
		print(f"Distribution of keys in the {t}. text:\n{get_key_mapping(array, keys)}")
		print(f"Keys appearing less than {min_occurrence} times in the {t}. text:\n{get_below_minimum_key_mapping(array, keys, min_occurrence)}")
		print(f"Unknown keys in the {t}. text: {unknown[t]}")

	print(f"Distribution of keys among all texts:\n{get_key_mapping(total_key_occurences, keys)}")

	print(f"Keys appearing less than {min_occurrence} times total among all texts:\n{get_below_minimum_key_mapping(total_key_occurences, keys, min_occurrence)}")

	print(f"Total amount of keys:\n{total_keys}")

	relative_occurances = [f"{(value/total_keys)*100:.2f} %" for value in total_key_occurences]
	print(f"Relative distribution of keys among all texts:\n{get_key_mapping(relative_occurances, keys)}")

	print(f"Total amount of letters:\n{total_letters}")
	if num_sharp_s:
		print(f"Total amount of letters with \"ß\":\n{total_letters + num_sharp_s}")

	print(f"Total amount of shifts: {num_shift}")

	relative_letter_occurances = [f"{(value/total_letters)*100:.2f} %" for value in total_letter_occurences]
	print(f"Relative distribution of letters among all texts:\n{get_key_mapping(relative_letter_occurances, letters)}")


def plot(plot_type, data, x, y, title, path, filename, png=False, font_scale=1, palette="Set1"):
	"""Plot a seaborn bar plot."""
	# reset font scale
	sns.set(font_scale=font_scale)
	sns.set_palette(palette)
	plt.figure()
	fig = sns.barplot(x=x, y=y, data=data)
	unique_labels = len(data[x].unique())
	# if x ticks contain strings, prevent overlapping
	if data[x].dtype == object:
		# reduce label size in case too many labels are given which are too wide
		if unique_labels >= 6 and any(len(str(label)) > 5 for label in data[x].values):
			if font_scale >= 1.4:
				fig.tick_params(axis="x", labelsize=9*font_scale)
		# adjust font size in case many small labels are given
		if unique_labels > 20:
			fig.tick_params(axis="x", labelsize=9*font_scale)
		if unique_labels > 30:
			fig.tick_params(axis="x", labelsize=8*font_scale)
		if unique_labels > 40:
			fig.tick_params(axis="x", labelsize=6*font_scale)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[:], labels=labels[:])
	if not any(fig.get_legend_handles_labels()):
		fig.get_legend().remove()
	if font_scale != 1:
		print("WARNING: You have changed the default font scale. Make sure to scale the image accordingly to receive a final scale of 1.")
	# show maximized for qt backend
	if matplotlib.get_backend().startswith('Qt'):
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
	# store
	if filename:
		if png:
			plt.savefig(os.path.join(path, filename + ".png"), dpi=300, bbox_inches="tight")
		else:
			plt.savefig(os.path.join(path, filename + ".svg"), dpi=300, bbox_inches="tight")
	else:
		plt.suptitle(title, fontsize=22)

def filename_from_title(title, filename=""):
	"""Auto-generate a filename from a given figure title (e.g. by replacing spaces)."""
	filename = "_".join(title.split(" ")).lower() + "_" + filename.split("/")[-1]
	filename = filename.replace("(", "").replace(")", "")
	return filename

def plot_data(keys, total_key_occurences, layout, show_plot, path, filename):
	# prepare data, leave out space and return
	total_x = total_key_occurences[:-2]
	labels = [f"{tpl[0]}\n{tpl[1]}" if tpl not in (" ", "\n") else "sp" if tpl == " " else "ret" for tpl in keys[:-2]]
	total_keys_no_space_return = sum(map(int, total_x))
	rel_x = [(value/total_keys_no_space_return) for value in total_key_occurences[:-2]]
	# sort
	rel_labels = [l for _, l in sorted(zip(rel_x,labels), reverse=True, key=lambda pair: pair[0])]
	rel_x = sorted(rel_x, reverse=True)
	total_labels = [l for _, l in sorted(zip(total_x,labels), reverse=True, key=lambda pair: pair[0])]
	total_x = sorted(total_x, reverse=True)
	# reshape
	x = "keys"
	y = f"relative frequency ({total_keys_no_space_return} keys total)"
	df_rel = pd.DataFrame({x: rel_labels, y: rel_x})
	title = f"Key Frequency ({layout})"
	# plot
	plot("bar", df_rel, x, y, title=title, path=path,
		filename=filename_from_title(title, filename) if filename else ""
	)
	# reshape
	y = f"absolute frequency ({total_keys_no_space_return} keys total)"
	df_total = pd.DataFrame({x: total_labels, y: total_x})
	title = f"Total Key Occurrence ({layout})"
	# plot
	plot("bar", df_total, x, y, title=title, path=path,
		filename=filename_from_title(title, filename) if filename else ""
	)
	if show_plot:
		plt.show()

def run(layout, filename, min_occurrence, ignore_header, path, save_plot):
	assert layout in ["de", "us"]
	keys, letters, key_occurences, unknown, num_sharp_s, num_shift, total_key_occurences, total_keys, total_letter_occurences, total_letters = get_data(layout, filename, ignore_header)
	print_data(keys, letters, key_occurences, unknown, num_sharp_s, num_shift, total_key_occurences, total_keys, total_letter_occurences, total_letters, min_occurrence)
	plot_data(keys, total_key_occurences, layout, plot, path, filename if save_plot else "")

	return keys, letters, key_occurences, unknown, num_sharp_s, num_shift, total_key_occurences, total_keys, total_letter_occurences, total_letters

def main(
		layout: "the keyboard layout to check on ('de' or 'us')",
		file_name: "the .key.csv file to analyze",
		min_occurrence: "the minimum number of key occurrences aimed at" = 10,
		ignore_header: "flag to ignore the first line (i.e. header) above each text extract" = True,
		save_plot: "save the plots" = False,
	):
	run(layout, file_name, min_occurrence, ignore_header, "results/analysis", save_plot)
