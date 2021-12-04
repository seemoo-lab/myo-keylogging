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
Plot the acceleration magnitudes of two Myos in the vicinity of a clap sync event.
"""

import pandas as pd

from . import utils as plot_utils
from preprocess import utils, fuse

def main(prefix="train-data/record-0.t1"):
	sync = utils.read_meta(prefix, rescale_imu=True)["sync"]
	left_imu = utils.read_sensor(f"{prefix}.left.imu.csv", rescale_imu=True)
	right_imu = utils.read_sensor(f"{prefix}.right.imu.csv", rescale_imu=True)

	_, point = fuse.find_joint_acc_lag(left_imu, right_imu, sync["min_acc_magnitude"], sync["max_lag"])

	left_imu = left_imu[(left_imu.index > point - 0.5) & (left_imu.index < point + 0.5)]
	right_imu = right_imu[(right_imu.index > point - 0.5) & (right_imu.index < point + 0.5)]

	left_imu[r"acceleration magnitude $\left[\dfrac{m}{s^2}\right]$"] = (left_imu.acc0**2 + left_imu.acc1**2 + left_imu.acc2**2)**0.5
	left_imu[r"time $[s]$"] = left_imu.index - (point - 0.5)
	right_imu[r"acceleration magnitude $\left[\dfrac{m}{s^2}\right]$"] = (right_imu.acc0**2 + right_imu.acc1**2 + right_imu.acc2**2)**0.5
	right_imu[r"time $[s]$"] = right_imu.index - (point - 0.5)

	left_imu["hue"] = "left hand"
	right_imu["hue"] = "right hand"
	merged = pd.concat([left_imu,right_imu])

	plot_utils.plotf(plot_utils.plot_clap_sync,
		data=merged,
		x=r"time $[s]$",
		y=r"acceleration magnitude $\left[\dfrac{m}{s^2}\right]$",
		hue="hue",
		style="hue",
		path="results/analysis", filename="clap_sync"
	)
