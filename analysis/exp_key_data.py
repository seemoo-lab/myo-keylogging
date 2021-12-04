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
Print the key data.
"""

import csv
import pathlib
import json
import random
random.seed(233)

import numpy as np
import pandas as pd

from preprocess.encode import KEYMAP_DE, KEYMAP_US

CONSIDERED_KEYS = [keysyms[0] for keysyms in KEYMAP_DE.values()]
CONSIDERED_KEYS += ["ssharp", "plus", "numbersign", "less", "comma", "period", "minus", "Caps_Lock", "udiaeresis", "odiaeresis", "adiaeresis"]
CONSIDERED_KEYS += [keysyms[0] for keysyms in KEYMAP_US.values()]
CONSIDERED_KEYS += ["equal", "bracketleft", "bracketright", "semicolon", "apostrophe", "slash"]
CONSIDERED_KEYS += ["ß", "+", "#", "<", ",", ".", "-", "ü", "ö", "ä", "=", "[", "]", ";", "'", "/"] # add replacement keys just to be sure
CONSIDERED_KEYS += list(map(str, range(10,23))) + list(map(str, range(24,37))) + list(map(str, range(38,49))) + ["94"] + list(map(str, range(50,63))) + ["65"] + ["66"] # add keycodes (Caps_Lock is 66)
KEYSYM_TO_SYMBOL = {"ssharp": "ß", "plus": "+", "numbersign": "#", "less": "<", "comma": ",", "period": ".", "minus": "-", "udiaeresis": "ü", "odiaeresis": "ö", "adiaeresis": "ä", "equal": "=", "bracketleft": "[", "bracketright": "]", "semicolon": ";", "apostrophe": "'", "slash": "/"}


def run(files, verbose=0):
	key_index = 2 # set to 1 for key codes and to 2 for key symbols

	# prepare dictionaries
	data = {
		"participant": [],
		"recording": [],
		"style": [],
		"layout": [],
		"task type": [],
		"time [min]": [],
		"keys per minute": [],
		"total number of keys": [],
		"total number of keys no auto": [],
		"total number of modifiers": [],
		"total number of backspaces": [],
		"total uncorrected errors": [], # levenshtein
		"total number of overlaps": [],
		"total number of non-modifier overlaps": []
	}

	keylevel_data = {
		"participant": [],
		"recording": [],
		"style": [],
		"layout": [],
		"task type": [],
		"task": [],
		"keysym": [],
		"keycode": [],
		"total amount overlapped": []
	}

	# get mistakes and backspace hits (and store the values in above dictionary)
	for task, filename in enumerate(files):

		# gather meta data
		with open(f"{str(filename)[:-7]}meta.json") as jsonfile:
			json_data = json.load(jsonfile)

		try:
			task_type = json_data["common"]["task_type"]
		except KeyError:
			task_type = "passwords"
		recording = json_data["id"]["collection"]
		participant = json_data["id"]["user"]
		kb_layout = json_data["keyboard"]["layout"]
		typing_style = " ".join(json_data["common"]["typing_style"].split("_")).capitalize()
		if not typing_style == "Touch typing":
			typing_style = "Non-touch typing"
		levenshtein = json_data["common"].get("levenshtein_distance", 0)
		num_typed_keys = json_data["common"]["num_given_keys_with_auto_repeat"]
		num_auto_repeat_keys = json_data["common"]["num_auto_repeat_keys"]
		num_true_keys = json_data["common"].get("num_true_keys", num_typed_keys)

		# gather key data
		missing_return = False
		# check for missing return
		if any([string.startswith("SYNC_ERROR_B") for string in json_data["notes"]]):
			missing_return = True
			if verbose:
				print(f"Handling missing second return at the start of: {filename}")

		# prepare variables
		num_backspaces = 0
		first = 0
		last = 0
		total_keys_within_task = 0
		full_key_presses = []
		full_key_presses_code = []
		amount_overlapped = []
		non_mod_amount_overlapped = []
		num_modifiers = 0

		with open(filename, newline="") as csvfile:
			csvreader = csv.reader(csvfile)
			start_count = 0
			pressed_keys = []
			for i, row in enumerate(csvreader):

				# get backspaces
				if row[2] == "BackSpace" and row[3] == "press":
					num_backspaces += 1

				# get modifiers
				if row[2] in ("Shift_L", "Shift_R") and row[3] == "press":
					num_modifiers += 1

				# get overlapping keys
				keys = [t[0] for t in pressed_keys]
				if row[3] == "press" and row[key_index] not in keys:
					# each key pressed is overlapped by all keys pressed before that are not released
					for i,_ in enumerate(pressed_keys):
						pressed_keys[i][1] += 1
					pressed_keys.append([row[key_index], 0])
				elif row[3] == "release" and row[key_index] in keys:
					index = keys.index(row[key_index])
					key = pressed_keys.pop(index)
					full_key_presses.append(key[0]) # key pressed
					assert key[0] == row[key_index]
					full_key_presses_code.append(row[1])
					amount_overlapped.append(key[1]) # number of keys overlapped
					non_mod_amount_overlapped.append(0 if key[0] in ("Shift_L", "Shift_R", "50", "62") else key[1])

				# get timings for all tasks
				if start_count > 2:
					if row[3] == "release": # only count releases to prevent counting auto-presses
						last = float(row[0])
						total_keys_within_task += 1
					continue
				# if the task is the game task, skip all introduction texts, the number of which varied
				# between data recordings
				# (approximate start time is set to the first number pressed, as starting a ship by
				# pressing a number is the first "reasonable" action ingame)
				if task_type == "game":
					if row[2].isdigit():
						first = float(row[0])
						start_count = 3
					continue
				# if the task is not the game task, wait for two enters pressed and skip all that comes
				# before or in between unless the missing_return flag is set in which case only one
				# enter needs to be found
				# this may be inaccurate should the first required (or typed) key be enter or should
				# enter be pressed accidentally before the clap sync was successful, followed by one
				# enter and one or more keys before the task starts with another enter press
				# but it is more accurate for other error patterns like too many enter presses that
				# should occur more often
				if start_count == 2 or start_count == 1 and missing_return:
					if row[1] == "36": # reset if enter is pressed again
						if verbose:
							print(f"Additional enter presses detected in: {filename}")
						start_count -= 1
					else: # start
						if i != 5 and verbose:
							print(f"-- Start at key {i//2+1} instead of 3.")
						first = float(row[0])
						start_count = 3
					continue
				# ignore everything before the first enter release
				if row[1] == "36" and row[3] == "release":
					start_count += 1
					continue
				# print warnings and information
				if row[1] not in ("36", "keycode"):
					if start_count == 1 and verbose:
						print(f"WARNING: Possibly missing second return at the start of: {filename}")
						print(f" or misclick if the key value is low.")
						#subprocess.Popen(f"sed -n 1,10p {filename}".split())
					if start_count == 0 and verbose:
						print(f"Extra characters found at the start of: {filename}")
				# do not continue here

		# sum overlaps
		overlaps = sum(1 if el > 0 else 0 for el in amount_overlapped)
		non_mod_overlaps = sum(1 if el > 0 else 0 for el in non_mod_amount_overlapped)

		# store data
		data["recording"].append(recording)
		data["participant"].append(participant)
		data["task type"].append(task_type)
		data["style"].append(typing_style)
		data["layout"].append(kb_layout)
		data["time [min]"].append((last - first)/60)
		data["keys per minute"].append(total_keys_within_task/((last - first)/60))
		data["total number of keys"].append(num_typed_keys)
		data["total number of keys no auto"].append(num_typed_keys - num_auto_repeat_keys) # can be != len(full_key_presses) due to non-released keys
		data["total number of modifiers"].append(num_modifiers)
		data["total number of backspaces"].append(num_backspaces)
		data["total uncorrected errors"].append(levenshtein)
		data["total number of overlaps"].append(overlaps)
		data["total number of non-modifier overlaps"].append(non_mod_overlaps)

		# store overlapping
		keylevel_data["task type"].extend([task_type for el in full_key_presses])#("\n".join(task_type.split(" ")))
		keylevel_data["task"].extend([task for el in full_key_presses])
		keylevel_data["keysym"].extend([el for el in full_key_presses])
		keylevel_data["keycode"].extend([el for el in full_key_presses_code])
		keylevel_data["total amount overlapped"].extend([el for el in amount_overlapped])
		keylevel_data["recording"].extend([recording for el in full_key_presses])
		keylevel_data["participant"].extend([participant for el in full_key_presses])
		keylevel_data["style"].extend([typing_style for el in full_key_presses])
		keylevel_data["layout"].extend([kb_layout for el in full_key_presses])
		#print(filename)
		#print(f"# keys typed: {num_typed_keys}")
		#print(f"# full key presses: {len(full_key_presses)}")
		#print(f"# keys typed no auto: {num_typed_keys-num_auto_repeat_keys}")
		#print(f"# overlaps: {overlaps}")
		#print(f"# non-shift overlaps: {non_mod_overlaps}")
		#print(f"amount overlapped: {sum(amount_overlapped)}")

	# create and sort dataframes to prevent plots sorted in a different way
	df_data = pd.DataFrame(data)
	df_data = df_data.sort_values(by=["task type"]).reset_index(drop=True)
	df_keylevel_data = pd.DataFrame(keylevel_data)
	df_keylevel_data = df_keylevel_data.sort_values(by=["task type"]).reset_index(drop=True)

	# add more columns
	# add task filters
	df_data["generalized task type"] = df_data["task type"]
	df_data["generalized task type"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform disappearing"}, regex=True, inplace=True)
	df_data["task category"] = df_data["task type"]
	df_data["task category"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform"}, regex=True, inplace=True)
	df_data["generalized task category"] = df_data["task type"]
	df_data["generalized task category"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform", "text": "text-based", "pangram": "text-based"}, regex=True, inplace=True)
	# add relative values
	#df_data["number of modifiers"] = df_data["total number of modifiers"]/df_data["total number of keys"]
	df_data["number of backspaces"] = df_data["total number of backspaces"]/df_data["total number of keys"]
	df_data["number of overlaps"] = df_data["total number of overlaps"]/df_data["total number of keys no auto"]
	df_data["number of non-modifier overlaps"] = df_data["total number of non-modifier overlaps"]/df_data["total number of keys no auto"]
	# prepare totals
	df_keylevel_data["total key frequency"] = 1 # add number of occurrences column
	df_keylevel_data["total overlap frequency"] = 1.0
	df_keylevel_data.loc[df_keylevel_data["total amount overlapped"] == 0, "total overlap frequency"] = 0.0
	# add task filters
	df_keylevel_data["generalized task type"] = df_keylevel_data["task type"]
	df_keylevel_data["generalized task type"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform disappearing"}, regex=True, inplace=True)
	df_keylevel_data["task category"] = df_keylevel_data["task type"]
	df_keylevel_data["task category"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform"}, regex=True, inplace=True)
	df_keylevel_data["generalized task category"] = df_keylevel_data["task type"]
	df_keylevel_data["generalized task category"].replace({"uniform \d": "uniform", "uniform disappearing \d": "uniform", "text": "text-based", "pangram": "text-based"}, regex=True, inplace=True)
	#with pd.option_context("display.max_rows", None, "display.max_columns", None):
	#	print(df_data)

	return df_data, df_keylevel_data


def main(
		path: "path to a directory to load data from" = "train-data/",
		verbose: "verbosity level" = 0
	):
	# get all key files
	files = pathlib.Path(path).glob("*.key.csv")

	# get the number of participants
	num_participants = len(list(pathlib.Path(path).glob("*t2.meta.json")))
	print(f"number of participants: {num_participants}")

	df_data, df_keylevel_data = run(files, verbose=verbose)

	print("")
	print(f"total hours of data taking: {df_data['time [min]'].sum()/60}")
	print(f"total time taken per recording:\n{df_data.groupby('recording')['time [min]'].agg(np.sum)}\n")
	print(f"mean time taken over all recordings:\n{df_data.groupby('recording')['time [min]'].agg(np.sum).mean()}\n")

	print(f"mean typing speed of recordings across all tasks:\n{df_data.groupby(['recording','participant'])['keys per minute'].agg(np.mean).sort_values()}\n")
	print(f"mean typing speed of recordings with a certain typing style across all tasks:\n{df_data.groupby('style')['keys per minute'].agg(np.mean)}\n")
	print(f"mean typing speed of recordings with a certain typing style across all TEXT tasks:\n{df_data.loc[df_data['task type'] == 'text'].groupby('style')['keys per minute'].agg(np.mean)}\n")
	print(f"mean typing speed of recordings with a certain typing style across all UNIFORM tasks:\n{df_data.loc[df_data['task type'].str.startswith('uniform')].groupby('style')['keys per minute'].agg(np.mean)}\n")

	print(f"mean typing speed of recordings per task type:\n{df_data.groupby('generalized task category')['keys per minute'].agg(np.mean)}\n")

	print(f"mean overlaps of recordings across all tasks:\n{df_data.groupby(['recording'])['number of non-modifier overlaps'].agg(np.mean)}\n")
	print(f"mean overlaps of all recordings:\n{np.mean(df_data['number of non-modifier overlaps'])}\n")
