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

from scipy import signal
import numpy as np
from . import utils

def notch(fs, f0, bandwidth=2):
	Q = f0 / bandwidth
	num, den = signal.iirnotch(f0, Q, fs=fs)
	return num, den

# follow the recommendation from De Luca, Carlo J., et al. "Filtering the surface EMG signal:
# Movement artifact and baseline noise contamination." Journal of biomechanics 43.8 (2010)
def highpass(fs, f0=10, order=2):
	num, den = signal.butter(order, f0, btype="highpass", analog=False, fs=fs)
	return num, den

def freq_response(func, fs, num_freqs=2048):
	num, den = func(fs)
	w, h = signal.freqz(num, den, worN=num_freqs)
	freq = w * fs / (2 * np.pi)
	return freq, 20 * np.log10(abs(h))

def apply_filters(func_list, sensor_data):
	fs = utils.get_frequency(sensor_data.index)
	for func in func_list:
		num, den = func(fs)
		for col_name in sensor_data:
			sensor_data[col_name] = signal.filtfilt(num, den, sensor_data[col_name])
	return sensor_data
