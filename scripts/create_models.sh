#!/bin/bash -x

# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2021  Matthias Gazzari, Annemarie Mattmann
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


# default binary:

python -m ml resnet &
python -m ml resnet11 &
python -m ml wavenet &
python -m ml crnn &

# binary sensor tests:

#e
python -m ml resnet --ignore "quat" "acc" "gyro" &
python -m ml resnet11 --ignore "quat" "acc" "gyro" &
python -m ml wavenet  --ignore "quat" "acc" "gyro" &
python -m ml crnn  --ignore "quat" "acc" "gyro" &

#g
python -m ml resnet --ignore "quat" "acc" "emg" &
python -m ml resnet11 --ignore "quat" "acc" "emg" &
python -m ml wavenet  --ignore "quat" "acc" "emg" &
python -m ml crnn  --ignore "quat" "acc" "emg" &

#a
python -m ml resnet --ignore "quat" "emg" "gyro" &
python -m ml resnet11 --ignore "quat" "emg" "gyro" &
python -m ml wavenet  --ignore "quat" "emg" "gyro" &
python -m ml crnn  --ignore "quat" "emg" "gyro" &

# default multiclass:

python -m ml resnet --encoding "multiclass" &
python -m ml resnet11 --encoding "multiclass" &
python -m ml wavenet --encoding "multiclass" &
python -m ml crnn --encoding "multiclass" &

# multiclass sensor tests:

#e
python -m ml resnet --encoding "multiclass" --ignore "quat" "acc" "gyro" &
python -m ml resnet11 --encoding "multiclass" --ignore "quat" "acc" "gyro" &
python -m ml wavenet --encoding "multiclass"  --ignore "quat" "acc" "gyro" &
python -m ml crnn --encoding "multiclass"  --ignore "quat" "acc" "gyro" &

#g
python -m ml resnet --encoding "multiclass" --ignore "quat" "acc" "emg" &
python -m ml resnet11 --encoding "multiclass" --ignore "quat" "acc" "emg" &
python -m ml wavenet --encoding "multiclass"  --ignore "quat" "acc" "emg" &
python -m ml crnn --encoding "multiclass"  --ignore "quat" "acc" "emg" &

#a
python -m ml resnet --encoding "multiclass" --ignore "quat" "emg" "gyro" &
python -m ml resnet11 --encoding "multiclass" --ignore "quat" "emg" "gyro" &
python -m ml wavenet --encoding "multiclass"  --ignore "quat" "emg" "gyro" &
python -m ml crnn --encoding "multiclass"  --ignore "quat" "emg" "gyro" &
