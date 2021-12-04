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
Plot the filter frequency responses.
"""

from functools import partial

from scipy import signal
from preprocess import filt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(fs=200):

	freq, response = filt.freq_response(partial(filt.notch, f0=50), fs)
	pd.DataFrame({"freq": freq, "response": response}).plot(x="freq", y="response")
	freq, response = filt.freq_response(filt.highpass, fs)
	pd.DataFrame({"freq": freq, "response": response}).plot(x="freq", y="response")
	plt.show()

if __name__ == "__main__":
	main()
