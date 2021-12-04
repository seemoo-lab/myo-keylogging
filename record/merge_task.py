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
import csv
import math

def get_files(prefix, task, num):
	for arm in ("left", "right"):
		for sensor in ("emg", "imu"):
			yield f"{prefix}-{num}.{task}.{arm}.{sensor}.csv"
	yield f"{prefix}-{num}.{task}.key.csv"

def merge_task(prefix, task, src, dest):
	# determine max timestamp
	offset = 0
	for dest_file in get_files(prefix, task, dest):
		with open(dest_file) as dest_obj:
			csv_reader = csv.reader(dest_obj, delimiter=",")
			offset = max(offset, float(next(reversed(list(csv_reader)))[0]))
	offset = math.ceil(offset) + 30
	print(offset)

	# append src files to dest files
	for dest_file, src_file in zip(get_files(prefix, task, dest), get_files(prefix, task, src)):
		with open(dest_file, "a") as dest_obj, open(src_file) as src_obj:
			csv_reader = csv.reader(src_obj, delimiter=",")
			csv_writer = csv.writer(dest_obj, delimiter=",")
			next(csv_reader) # skip header
			for row in csv_reader:
				csv_writer.writerow([float(row[0]) + offset] + row[1:])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prefix", help="the file prefix of csv files to be merged")
	parser.add_argument("--task", help="the file prefix of csv files to be merged")
	parser.add_argument("--dest", type=int, help="the destination prefix number")
	parser.add_argument("--src", type=int, help="the source prefix number")
	args = parser.parse_args()
	merge_task(args.prefix, args.task, args.src, args.dest)

if __name__ == "__main__":
	main()
