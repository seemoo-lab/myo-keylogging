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

import argparse
import glob
import csv
import json
import collections

import german_keymap
import english_keymap
import utils

def print_warning(message):
	print(f"\x1b[38;5;160mWARNING: {message}\x1b[0m")

# flatten a nested array with multiple different dimensions
def flatten(lst):
	texts = []
	for el in lst:
		if not isinstance(el, str):
			texts.extend(flatten(el))
		else:
			texts.append(el)
	return texts

def sanitize(string, key_layout):
	# take care of shift uppercase (result is layout-dependent)
	uppercase = False
	case_count = 0
	for i, char in enumerate(string):
		if char == chr(752):
			uppercase = True
			case_count += 1
			continue
		if char == chr(751):
			uppercase = False
			case_count -= 1
			continue
		# ignore space, newline, backspace and all ignored characters
		if char in (" ", "\n", chr(767), chr(0x25a0)):
			continue
		if uppercase:
			try:
				string = string[:i] + key_layout[char] + string[i+1:]
			except KeyError: # a keyerror may happen for <
				string = string[:i] + chr(0x25a0) + string[i+1:]
	string = string.replace(chr(751), "")
	string = string.replace(chr(752), "")

	# remove ignored characters
	string = string.replace(chr(0x25a0), "")

	# handle backspace
	i = 0
	while i < len(string):
		if string[i] == chr(767):
			if i > 0 and string[i - 1] != "\n": # only delete chars within a prompt
				string = string[:i-1] + string[i+1:]
				i -= 1
			else:
				string = string[:i] + string[i+1:]
		else:
			i += 1

	if case_count != 0:
		print_warning(f"The number of case modifiers does not match (off by {abs(case_count)} with additional {chr(752) if case_count > 0 else chr(751)}).")
	return string

def interpret(string, repetitions, task_type, reps, pw1, pw2): # only works for 2 pws currently
	# keep content after the first two line breaks (or three in case of a rare input issue) ...
	start_offset = 3 if string.startswith("\n" * 3) else 2
	# ... and before the last two or three line breaks (depending on the task type)
	end_offset = -2 if task_type =="insecure" else -3
	pw_list = string.split("\n")[start_offset:end_offset]
	assert pw_list[-1] == pw2
	print(pw_list)

	# find index of the "reps" occurrence of the first password to split the pw list into two parts
	# note that every second index is ignored to not count "correct" passwords typed in between 
	pos_of_correct_pw1s = [n*2 for n, pw in enumerate(pw_list[::2]) if pw == pw1]
	left_split_idx = pos_of_correct_pw1s[reps - 1] + 1

	# determine left password list
	left_pw_list = pw_list[0:left_split_idx]
	left_pw_list = left_pw_list[::-2][::-1]
	print("left", left_pw_list)
	assert left_pw_list[-1] == pw1
	assert left_pw_list.count(pw1) == 4

	# find the starting position for the right split by taking the task type into account
	# by adding either one or two to the slice position used for the left password list
	right_split_idx = left_split_idx + (1 if task_type =="insecure" else 2)

	# determine the right password list
	right_pw_list = pw_list[right_split_idx:]
	right_pw_list = right_pw_list[::-2][::-1]
	print("right", right_pw_list)
	assert right_pw_list[-1] == pw2
	assert right_pw_list.count(pw2) == 4

	return len(left_pw_list) , len(right_pw_list)

# read keys from one participant
def process(files, de_keys, us_keys):
	files.sort()

	with open(f"{files[0][:-7]}meta.json") as jsonfile:
		json_data = json.load(jsonfile)
	print(f"participant: {json_data['id']['user']}")
	layout = "english" if json_data["keyboard"]["layout"] == "us" else "german"
	print(f"layout: {layout}")
	key_layout = us_keys if layout == "english" else de_keys

	print("-"*70)
	for filename in files:
		meta_file = f"{filename[:-7]}meta.json"
		with open(meta_file) as jsonfile:
			json_data = json.load(jsonfile)
		#task_type = json_data["common"]["task_type"]

		print(f"task: {filename.split('/')[-1]}")
		# get the number of pressed keys and the post-processed input
		total_tries = 0
		total_keys = 0
		auto_repeated = 0
		user_input = ""
		with open(filename, newline="") as csvfile:
			csvreader = csv.reader(csvfile)
			pressed_keys = set([])
			caps_lock_active = False
			for row in csvreader:
				if row[3] == "press":
					keysym = row[2]
					if keysym in pressed_keys:
						auto_repeated += 1
					# transform caps lock to shift and handle it as such
					if keysym == "Caps_Lock":
						# ignore second caps lock press
						if caps_lock_active and not keysym in pressed_keys:
							total_keys += 1
							continue
					user_input += utils.convert_to_char(keysym)
					pressed_keys.add(keysym)
					total_keys += 1
				elif row[3] == "release":
					if row[2] in ("Shift_L", "Shift_R"):
						user_input += utils.convert_to_char("Shift_Release")
					elif row[2] == "Caps_Lock":
						if caps_lock_active:
							user_input += utils.convert_to_char("Shift_Release")
							caps_lock_active = False
						else:
							caps_lock_active = True
					pressed_keys.discard(row[2])
		# print the key presses truly typed (lower-case)
		print(user_input)
		# handle shift and Caps_Lock
		sanitized_input = sanitize(user_input, key_layout)
		print(f"given: {total_keys} keys with {auto_repeated} auto-repeated keys")

		# prepare json storage
		json_data["common"]["num_given_keys_with_auto_repeat"] = total_keys
		json_data["common"]["num_auto_repeat_keys"] = auto_repeated

		# get the keys required by the task (ground truth)
		print(sanitized_input)
		reps = json_data["task"]["repetitions"]
		task_type = json_data["common"]["task_type"]
		pw1 = json_data["task"]["passwords"][0]
		pw2 = json_data["task"]["passwords"][1]
		attempts_first_part, attempts_second_part = interpret(sanitized_input, reps, task_type, reps, pw1, pw2)
		json_data["task"]["attempts"] = [attempts_first_part, attempts_second_part]

		# store results in meta data
		answer = input(f"Store the following to {meta_file}:\n{json_data['common']}\n{json_data['task']}? [Y|n]")
		if answer != "n":
			print("stored")
			json.dump(json_data, open(meta_file, "w"), indent=4, ensure_ascii=False)
		print("-"*70)
	print("-"*70)

def main():
	parser = argparse.ArgumentParser(description="Read user input")
	parser.add_argument("-p", "--path", default="../test-data/", type=str, help="standard path to search for files")
	parser.add_argument("-f", "--files", default=None, nargs="+", type=str, help="the key csv files to read - if this is not given, all files in PATH will be read")
	args = parser.parse_args()

	de_keys = dict((k, v) for k, v in german_keymap.keys)
	us_keys = dict((k, v) for k, v in english_keymap.keys)

	if args.files:
		files = args.files
		process(files, de_keys, us_keys)
	else:
		# get prefix of all participants
		files = glob.glob(f"{args.path}*t0.key.csv")
		print(f"{len(files)} participants")
		files.sort()
		# get key csv files for all tasks for each participant successively
		for name in files:
			name = name[:-10]
			process(glob.glob(f"{name}*.key.csv"), de_keys, us_keys)

if __name__ == "__main__":
	main()
