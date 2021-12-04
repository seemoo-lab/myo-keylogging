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

import pathlib
import json

merged = {}

for file_name in sorted(pathlib.Path("results/ml").glob("*.result.json")):
	with open(file_name) as fp:
		print(f"File: {file_name}")
		json_obj = json.load(fp)
		# remove package versions
		json_obj.pop("pkg_versions")
		# set unique run ID, number of iterations and number of samples to common values to avoid unnecessary mismatches
		json_obj["params"]["uid"] = 0
		json_obj["params"]["n_iter"] = 0
		json_obj["run_config"]["n_samples"] = -1
		json_obj["run_config"]["estimator"] = None
		# add seg_width parameter to run parameters
		seg_width = json_obj["params"].pop("seg_width")
		for i in range(len(json_obj["results"]["hpo"])):
			json_obj["results"]["hpo"][i]["params"]["seg_width"] = seg_width
		# create unique tag
		tag = "_".join(json_obj["info"].values())
		if tag not in merged.keys():
			merged[tag] = json_obj
		elif all(merged[tag][key] == json_obj[key] for key in ("info", "params", "run_config")):
			merged[tag]["results"]["hpo"].extend(json_obj["results"]["hpo"])
		else:
			print(list(merged[tag][key] == json_obj[key] for key in ("info", "params", "run_config")))
			raise ValueError

for tag, single_merged in merged.items():
	single_merged["params"]["n_iter"] = len(single_merged["results"]["hpo"])
	with open(f"results/ml/merged_{tag}.result.json", "w") as fp:
		json.dump(single_merged, fp, indent=4)
