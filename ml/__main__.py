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

# filter sklearn UndefinedMetricWarning warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

import logging
LOG = logging.getLogger(__name__)

import auto_load

if __name__ == "__main__":
	auto_load.execute(__file__, __package__)
