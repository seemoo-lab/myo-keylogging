#!/bin/bash -x

# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2020-2021  Matthias Gazzari, Annemarie Mattmann
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

# binary hpo:

python -m ml resnet --func shallow_hpo --step 5 --n_iter 25 --uid 0 &> results/ml/resnet_0.log &
python -m ml wavenet --func shallow_hpo --step 5 --n_iter 25 --uid 0 &> results/ml/wavenet_0.log &
python -m ml crnn --func shallow_hpo --step 5 --n_iter 25 --uid 0 &> results/ml/crnn_0.log &

python -m ml resnet --func shallow_hpo --step 5 --n_iter 25 --uid 1 &> results/ml/resnet_1.log &
python -m ml wavenet --func shallow_hpo --step 5 --n_iter 25 --uid 1 &> results/ml/wavenet_1.log &
python -m ml crnn --func shallow_hpo --step 5 --n_iter 25 --uid 1 &> results/ml/crnn_1.log &

python -m ml resnet --func shallow_hpo --step 5 --n_iter 25 --uid 2 &> results/ml/resnet_2.log &
python -m ml wavenet --func shallow_hpo --step 5 --n_iter 25 --uid 2 &> results/ml/wavenet_2.log &
python -m ml crnn --func shallow_hpo --step 5 --n_iter 25 --uid 2 &> results/ml/crnn_2.log &

# multiclass hpo:

python -m ml resnet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 0 &> results/ml/resnet_0.log &
python -m ml wavenet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 0 &> results/ml/wavenet_0.log &
python -m ml crnn --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 0 &> results/ml/crnn_0.log &

python -m ml resnet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 1 &> results/ml/resnet_1.log &
python -m ml wavenet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 1 &> results/ml/wavenet_1.log &
python -m ml crnn --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 1 &> results/ml/crnn_1.log &

python -m ml resnet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 2 &> results/ml/resnet_2.log &
python -m ml wavenet --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 2 &> results/ml/wavenet_2.log &
python -m ml crnn --encoding multiclass --func shallow_hpo --step 1 --n_iter 25 --uid 2 &> results/ml/crnn_2.log &
