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
Show all keys pressed inside a given .key.csv file
"""

from preprocess import utils

def main(file_name: "the file name of the .key.csv file" = "train-data/record-0.t2.key.csv"):

	key_data = utils.read_csv(file_name, index_col=0)
	key_data = key_data[key_data.event == "press"]
	key_data = key_data.replace({
		"Return": " Return ",
		"Shift_L": " Shift_L ",
		"Shift_R": " Shift_R ",
		"space": " ",
		"adiaeresis": "ä",
		"odiaeresis": "ö",
		"udiaeresis": "ü",
		"ssharp": "ß",
		"plus": "+",
		"minus": "-",
		"less": "<",
		"greater": ">",
		"period": ".",
		"comma": ",",
	})
	print(key_data)
	print("".join(key_data.keysym))
