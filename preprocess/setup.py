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


"""A Python library to preprocess the data recorded during the Myo armband data study."""

import setuptools

def read_file(file_name):
	"""Return the content of a file."""
	with open(file_name) as file_obj:
		return file_obj.read()

setuptools.setup(
	name="preprocess",
	description=__doc__,
	long_description=read_file("README.rst"),
	author="Matthias Gazzari, Annemarie Mattmann",
	packages=setuptools.find_packages(),
	# requirements
	install_requires=[
		"pandas>=0.23.4",
		"numpy>=1.15.4",
		"scipy>=1.1.0",
		"pytest>=3.6.2",
		"joblib>=0.13.1",
	],
	python_requires=">=3.8",
	# further description
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Science/Research",
		"Topic :: Software Development :: Libraries :: Python Modules",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
	],
)
