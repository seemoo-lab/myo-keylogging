#!/bin/bash -x

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

rm results/train-data-skew.csv

TRAIN_DATA_CMD="python -m analysis exp_data_imbalance --store --path train-data/ --csv_file results/train-data-skew.csv"

rm -r cache

for USER in {0..50}
do
	${TRAIN_DATA_CMD} --user ${USER}
	${TRAIN_DATA_CMD} --user ${USER} --task_types "text"
	${TRAIN_DATA_CMD} --user ${USER} --task_types "uniform"
	${TRAIN_DATA_CMD} --user ${USER} --task_types "game"
	${TRAIN_DATA_CMD} --user ${USER} --task_types "pangram"
done

rm -r cache

${TRAIN_DATA_CMD}
${TRAIN_DATA_CMD} --task_types "text"
${TRAIN_DATA_CMD} --task_types "uniform"
${TRAIN_DATA_CMD} --task_types "game"
${TRAIN_DATA_CMD} --task_types "pangram"

rm results/test-data-skew.csv

TEST_DATA_CMD="python -m analysis exp_data_imbalance --store --path test-data/ --csv_file results/test-data-skew.csv --cut all"

rm -r cache

for USER in {0..50}
do
	echo "Run with ${USER}"
	${TEST_DATA_CMD} --user ${USER}
	${TEST_DATA_CMD} --user ${USER} --task_types "insecure"
	${TEST_DATA_CMD} --user ${USER} --task_types "pwgen"
	${TEST_DATA_CMD} --user ${USER} --task_types "xkcd"
	${TEST_DATA_CMD} --user ${USER} --task_types "random"
done

rm -r cache

${TEST_DATA_CMD}
${TEST_DATA_CMD} --task_types "insecure"
${TEST_DATA_CMD} --task_types "pwgen"
${TEST_DATA_CMD} --task_types "xkcd"
${TEST_DATA_CMD} --task_types "random"

rm -r cache

for USER in {0..50}
do
	echo "Run with ${USER}"
	${TEST_DATA_CMD} --user ${USER} --cut "last"
	${TEST_DATA_CMD} --user ${USER} --task_types "insecure" --cut "last"
	${TEST_DATA_CMD} --user ${USER} --task_types "pwgen" --cut "last"
	${TEST_DATA_CMD} --user ${USER} --task_types "xkcd" --cut "last"
	${TEST_DATA_CMD} --user ${USER} --task_types "random" --cut "last"
done

rm -r cache

${TEST_DATA_CMD} --cut "last"
${TEST_DATA_CMD} --task_types "insecure" --cut "last"
${TEST_DATA_CMD} --task_types "pwgen" --cut "last"
${TEST_DATA_CMD} --task_types "xkcd" --cut "last"
${TEST_DATA_CMD} --task_types "random" --cut "last"

# only to check the amount of keystrokes
#${TRAIN_DATA_CMD} --preserve_all
#${TEST_DATA_CMD} --preserve_all
#${TEST_DATA_CMD} --preserve_all --cut None
