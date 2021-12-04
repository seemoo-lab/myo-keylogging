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
import json
import numpy as np
import preprocess

irrelevant_keys = [50, 65, 62, 36, 22] # Shift_L, space, Shift_R, Return, BackSpace

def main():
	parser = argparse.ArgumentParser(description="Update cuts to not include return")
	parser.add_argument("-p", "--path", default="../test-data/", type=str, help="standard path to search for files")
	args = parser.parse_args()

	files = glob.glob(f"{args.path}*.key.csv")

	skipped_files = []
	for name in files:
		name = name[:-8]
		print(f"Handling file: {name}")
		key_data = preprocess.utils.read_csv(f"{name}.key.csv", index_col=0)
		meta_file = name + ".meta.json"
		with open(meta_file) as file_obj:
			meta_data = json.load(file_obj)

		cuts = preprocess.utils.get_cut_indices(key_data, meta_data, "all")
		print(cuts)
		key_data_array_ready = key_data.drop(["keysym", "event"], axis=1)
		slices, _ = preprocess.utils.slice_data_to_word((key_data_array_ready.values,), (cuts,))
		print("original slices:")
		for s in slices:
			print(np.transpose(s))

		# get new cut indices based on return keys within the slices produced by cutting
		print("Positions of return keys and cuts:")
		i = 0
		for word in slices:
			offset = 1
			return_rows = np.where(word == 36)[0]
			print(return_rows)
#			# separate first set or return clicks (before pw entry) from second set (after pw entry)
#			# WARNING: assumes that there are no wrong keys in between first set returns
#			for idx in range(len(return_rows)-1):
#				if return_rows[idx+1] - return_rows[idx] > 2:
#					offset = idx
#					break
			largest_distance = 0
			# separate first set or return clicks (before pw entry) from second set (after pw entry)
			# sometimes there is bullshit data in between two return hits of the first return set
			# e.g.:
			#
			# *shift clicks only* # password should be here
			# ehg_rt3m
			#
			# thus, the beginning of the password is determined by the largest distance between two
			# return clicks visible in the slice
			for idx in range(1, len(return_rows)-1, 2):
				if return_rows[idx+1] - return_rows[idx] > largest_distance:
					offset = idx
					largest_distance = return_rows[idx+1] - return_rows[idx]
			if offset > 3:
				print("Found garbage data in the beginning, trying to skip it and find the correct word...")
				print(np.transpose(word))
			elif offset > 1:
				print("More than one return key in the beginning.")
			# first cut after return release (first key to include)
			cuts[i] += return_rows[offset] + 1
			# second cut at return press (after last key to include)
			cuts[i+1] = cuts[i] + return_rows[offset+1] - return_rows[offset] - 1
			print(cuts)
			i += 2
		slices, _ = preprocess.utils.slice_data_to_word((key_data.values,), (cuts,))
		print(slices[0])

		# get time of narrower cuts
		new_cut_times = []
		for cut in cuts:
			new_cut_times.append(key_data.index[cut-1] + ((key_data.index[cut] - key_data.index[cut-1]) / 2))

		print(new_cut_times)

		# produce slices with updated cuts (to detect garbage data in narrowed down data)
		test_meta = {"cuts": new_cut_times, "task": meta_data["task"]}
		updated_cuts = preprocess.utils.get_cut_indices_between(key_data, test_meta, "all")
		updated_slices, _ = preprocess.utils.slice_data_to_word((key_data_array_ready.values,), (updated_cuts,))

		# delete empty cuts resulting from empty user entries
		to_be_deleted = []
		i = 0
		for idx in range(0, len(new_cut_times)-1, 2):
			if new_cut_times[idx] == new_cut_times[idx+1]:
				to_be_deleted.append(idx)
				to_be_deleted.append(idx+1)
				print(f"Marking {new_cut_times[idx]} and {new_cut_times[idx+1]} for deletion from cuts as they are identical.")
			elif np.unique(updated_slices[i]).size <= 1 and updated_slices[i][0][0] in irrelevant_keys:
				to_be_deleted.append(idx)
				to_be_deleted.append(idx+1)
				print(f"Marking {new_cut_times[idx]} and {new_cut_times[idx+1]} for deletion from cuts as the respective slice {np.transpose(updated_slices[i])} is empty or contains only a single unique entry.")
			i += 1
		if to_be_deleted != []:
			answer = input(f"INFORMATION: Deleting {len(to_be_deleted)//2} entries from cuts in {meta_file} as they contain identical pairs or garbage data. This may happen due to empty user entries (hitting return twice without typing in-between). Is this okay? [Y|n]")
			if answer == "n":
				print(f"Skipping file {meta_file}")
				skipped_files.append(meta_file)
				continue
			for idx in reversed(to_be_deleted):
				new_cut_times.pop(idx)
		else:
			print("Nothing deleted, everything looks fine so far.")

		# test newly produced slices
		test_meta = {"cuts": new_cut_times, "task": meta_data["task"]}
		#print(meta_data)
		print(test_meta)
		test_cuts = preprocess.utils.get_cut_indices_between(key_data, test_meta, "all")
		print(test_cuts)
		test_slices, _ = preprocess.utils.slice_data_to_word((key_data_array_ready.values,), (test_cuts,))
		print("updated slices:")
		for s in test_slices:
			print(np.transpose(s))

		print(test_cuts[0] - 1)
		print(test_cuts[-1] + 1)
		print(key_data_array_ready.values[test_cuts[0] - 1][0])
		print(key_data_array_ready.values[test_cuts[-1] + 1][0])

		# safety check (check if no part of the password is deleted)
		if key_data_array_ready.values[test_cuts[0] - 1][0] != 36:
			answer = input(f"WARNING: Too much deleted in {meta_file}:\n Key prior to first cut is not return!{key_data_array_ready.values[test_cuts[0] - 1][0]}.\n Skip this? [Y|n]")
			if answer != "n":
				print(f"Skipping file {meta_file}")
				skipped_files.append(meta_file)
				continue
		try:
			if key_data_array_ready.values[test_cuts[-1] + 1][0] != 36:
				answer = input(f"WARNING: Too much deleted in {meta_file}:\n Key after last cut is not return!{key_data_array_ready.values[test_cuts[-1] + 1][0]}.\n Skip this? [Y|n]")
				if answer != "n":
					print(f"Skipping file {meta_file}")
					skipped_files.append(meta_file)
					continue
		except KeyError:
			answer = input(f"WARNING: A key exception was raised (i.e. no key was skipped at the end of the last cut) in {meta_file}. This should not happen.\n Skip this? [Y|n]")
			if answer != "n":
				print(f"Skipping file {meta_file}")
				skipped_files.append(meta_file)
				continue

		# sanity check (check if return is indeed missing in the password)
		for pw in test_slices:
			for el in pw:
				if el[0] == 36:
					answer = input(f"WARNING: A return key remains in {meta_file}. This should not happen.\n Skip this? [Y|n]")
					if answer != "n":
						print(f"Skipping file {meta_file}")
						skipped_files.append(meta_file)
						continue

		# update meta data
		meta_data["cuts"] = new_cut_times
		json.dump(meta_data, open(meta_file, "w"), indent=4, ensure_ascii=False)

	print("Skipped files:")
	print(skipped_files)
	if skipped_files == []:
		print("No skipped files, seems like everything went well.")

if __name__ == "__main__":
	main()
