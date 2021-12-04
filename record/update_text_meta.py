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
import difflib

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
			string = string[:i] + key_layout[char] + string[i+1:]
	string = string.replace(chr(751), "")
	string = string.replace(chr(752), "")

	if case_count != 0:
		print_warning(f"The number of case modifiers does not match (off by {abs(case_count)} with additional {chr(752) if case_count > 0 else chr(751)}).")
	return string

def levenshtein_distance(given_sentence, org_sentence):
	dist = [[0] * (len(org_sentence)+1) for _ in range(len(given_sentence)+1)]

	for i in range(len(given_sentence)+1):
		dist[i][0] = i
	for i in range(len(org_sentence)+1):
		dist[0][i] = i

	for i in range(1, len(given_sentence)+1):
		for j in range(1, len(org_sentence)+1):
			if given_sentence[i-1] == org_sentence[j-1]:
				cost = 0
			else:
				cost = 1
			dist[i][j] = min(dist[i-1][j] + 1, dist[i][j-1] + 1, dist[i-1][j-1] + cost)
		#print("\n".join([" ".join([str(cell) for cell in row]) for row in dist]))
		#print("")
	return dist[-1][-1]

# find a sentence closest to a given sentence for two arrays of sentences and return the found
# sentences' combined length
def compare_to_original(given, original, key_layout, return_count=2):
	work_original = original.copy()
	given = given.split("\n")
	given = list(filter(None, given)) # remove empty strings
	print("")
	# apply backspace for better comparison
	for s, given_sentence in enumerate(given):
		c = 0
		wrong_break = 0
		while c < len(given_sentence):
			if given_sentence[c] == chr(767):
				given_sentence = given_sentence[:max(0, c-1)] + given_sentence[c+1:]
				c = c-1
				if c < 0:
					wrong_break += 1
					c = 0
			else:
				c += 1
		if wrong_break:
			print_warning("Wrong break detected.\n")
			given[max(0, s-1)] = given[max(0, s-1)][:-wrong_break] + given_sentence
			given[s] = ""
			wrong_break = 0
		else:
			given[s] = given_sentence
	given = list(filter(None, given)) # remove empty strings
	print("corrected to:")
	for sentence in given:
		print(sentence)
	print("")
	# reconstruct original
	reconstruction = []
	for sentence in given:
		for word in sentence.split(" "):
			new_original = []
			for org_sentence in work_original:
				if difflib.get_close_matches(word, org_sentence.split(" ")):
					new_original.append(org_sentence)
			work_original = new_original
			if len(work_original) == 1:
				break
			if not work_original: # retry if no close match was found
				work_original = original.copy()
		if len(work_original) != 1:
			work_original = ["WARNING: No match found!"]
		reconstruction.extend(work_original)
		work_original = original.copy()
	print("closest to:")
	total_keys = 2 # initial return
	for sentence in reconstruction:
		if sentence == "WARNING: No match found!":
			print_warning("No match found!")
			continue
		print(sentence)
		total_keys += len(sentence) + return_count # count return used to continue
		# CAREFUL: not always two returns were required for continue
		# add one key press for shift (but not if shift is required for successive keys)
		last_char = None
		for char in sentence:
			if char in key_layout.values() and not last_char in key_layout.values():
				total_keys += 1
			last_char = char
	# compute levenshtein distance
	l_dist = 0
	for s, given_sentence in enumerate(given):
		if reconstruction[s] == "WARNING: No match found!":
			print_warning("Leaving out computation for missing match!")
			continue
		l_dist += levenshtein_distance(given_sentence, reconstruction[s])
	print("")
	print(f"ground truth: {total_keys} keys total")
	print(f"Levenshtein distance: {l_dist}")
	return total_keys, l_dist

# read keys from one participant
def process(files, german_texts, english_texts, de_keys, us_keys):
	files.sort()

	with open(f"{files[0][:-7]}meta.json") as jsonfile:
		json_data = json.load(jsonfile)
	layout = "english" if json_data["keyboard"]["layout"] == "us" else "german"
	print(f"participant: {json_data['common']['tag'].split(' ')[-1]}")
	print(f"layout: {layout}")
	texts = english_texts if layout == "english" else german_texts
	key_layout = us_keys if layout == "english" else de_keys

	print("-"*70)
	for filename in files:
		meta_file = f"{filename[:-7]}meta.json"
		with open(meta_file) as jsonfile:
			json_data = json.load(jsonfile)
		task_type = json_data["common"]["task_type"]

		print(f"task: {filename.split('/')[-1]}")
		# get the number of pressed keys and the post-processed input
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
		# define default values as the game does not require a certain amount of key presses
		truth = -1
		l_dist = -1
		if task_type != "game":
			truth = 0
			if not task_type.startswith("uniform"):
				return_count = 2
				# check for different return requirement
				if any([string.startswith("NO_RETURN_SAFEGUARD") for string in json_data["notes"]]):
					print("corrected return count")
					return_count = 1
				truth, l_dist = compare_to_original(sanitized_input, texts, key_layout, return_count)
			else:
				print("")
				truth = 49*2+1 if layout == "english" else 50*2+1 # +1 for the first sync return
				# account for missing-shift-bug
				if any([string.startswith("MISSING_SHIFT") for string in json_data["notes"]]):
					truth -= 1
				l_dist = len(json_data["mistakes"]) # as only the non-corrected mistakes are counted
				print(f"ground truth: {truth} keys total")
			print("")

		# prepare json storage
		json_data["common"]["num_true_keys"] = truth
		json_data["common"]["levenshtein_distance"] = l_dist

		# store results in meta data
		answer = input(f"Store the following to {meta_file}:\n{json_data['common']}\n? [Y|n]")
		if answer != "n":
			print("stored")
			json.dump(json_data, open(meta_file, "w"), indent=4, ensure_ascii=False)
		print("-"*70)
	print("-"*70)

def main():
	parser = argparse.ArgumentParser(description="Read user input")
	parser.add_argument("-p", "--path", default="../train-data/", type=str, help="standard path to search for files")
	parser.add_argument("-f", "--files", default=None, nargs="+", type=str, help="the key csv files to read - if this is not given, all files in PATH will be read")
	args = parser.parse_args()

	de_keys = dict((k, v) for k, v in german_keymap.keys)
	us_keys = dict((k, v) for k, v in english_keymap.keys)

	german_text_dict = utils.load_texts("german")
	german_texts = flatten(list(german_text_dict.values()))
	english_text_dict = utils.load_texts("english")
	english_texts = flatten(list(english_text_dict.values()))

	if args.files:
		files = args.files
		process(files, german_texts, english_texts, de_keys, us_keys)
	else:
		files = glob.glob(f"{args.path}*t2.key.csv")
		print(f"{len(files)} participants")
		files.sort()
		# get key csv files for all tasks for each participant successively
		for name in files:
			name = name[:-10]
			process(glob.glob(f"{name}*.key.csv"), german_texts, english_texts, de_keys, us_keys)

if __name__ == "__main__":
	main()
