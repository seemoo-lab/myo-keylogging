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

# Run models on test data and create prediction pickles, performance logs (and plots)

bin_r="resnet_fit_all_gptu_binary_press_eag_"
bin_r11="resnet11_fit_all_gptu_binary_press_eag_"
bin_c="crnn_fit_all_gptu_binary_press_eag_"
bin_w="wavenet_fit_all_gptu_binary_press_eag_"

mul_r="resnet_fit_all_gptu_multiclass_press_eag_"
mul_r11="resnet11_fit_all_gptu_multiclass_press_eag_"
mul_c="crnn_fit_all_gptu_multiclass_press_eag_"
mul_w="wavenet_fit_all_gptu_multiclass_press_eag_"


# default binary:
python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default --save_only --basenames $bin_r $bin_r11 $bin_c $bin_w
# default multiclass:
python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default --save_only --basenames $mul_r $mul_r11 $mul_c $mul_w
# default chain run:
python -m analysis apply_models --model_path results/chain_default --encoding chain --data_path test-data/ --save_path results/results_chain_default --save_only --basenames $bin_r $bin_r11 $bin_c $bin_w $mul_r $mul_r11 $mul_c $mul_w --tolerance 10

# sensor tests binary, change names for models created by create_models.sh:
# e
python -m analysis apply_models --model_path results/bin_sensortest --encoding binary --data_path test-data/ --save_path results/results_bin_sensortest/emg --save_only --basenames resnet_fit_all_gptu_binary_press_e_ resnet11_fit_all_gptu_binary_press_e_ crnn_fit_all_gptu_binary_press_e_ wavenet_fit_all_gptu_binary_press_e_
# g
python -m analysis apply_models --model_path results/bin_sensortest --encoding binary --data_path test-data/ --save_path results/results_bin_sensortest/gyro --save_only --basenames resnet_fit_all_gptu_binary_press_g_ resnet11_fit_all_gptu_binary_press_g_ crnn_fit_all_gptu_binary_press_g_ wavenet_fit_all_gptu_binary_press_g_
# a
python -m analysis apply_models --model_path results/bin_sensortest --encoding binary --data_path test-data/ --save_path results/results_bin_sensortest/acc --save_only --basenames resnet_fit_all_gptu_binary_press_a_ resnet11_fit_all_gptu_binary_press_a_ crnn_fit_all_gptu_binary_press_a_ wavenet_fit_all_gptu_binary_press_a_

# sensor tests multiclass, change names for models created by create_models.sh:
# e
python -m analysis apply_models --model_path results/mc_sensortest --encoding multiclass --data_path test-data/ --save_path results/results_mc_sensortest/emg --save_only --basenames resnet_fit_all_gptu_multiclass_press_e_ resnet11_fit_all_gptu_multiclass_press_e_ crnn_fit_all_gptu_multiclass_press_e_ wavenet_fit_all_gptu_multiclass_press_e_
# g
python -m analysis apply_models --model_path results/mc_sensortest --encoding multiclass --data_path test-data/ --save_path results/results_mc_sensortest/gyro --save_only --basenames resnet_fit_all_gptu_multiclass_press_g_ resnet11_fit_all_gptu_multiclass_press_g_ crnn_fit_all_gptu_multiclass_press_g_ wavenet_fit_all_gptu_multiclass_press_g_
# a
python -m analysis apply_models --model_path results/mc_sensortest --encoding multiclass --data_path test-data/ --save_path results/results_mc_sensortest/acc --save_only --basenames resnet_fit_all_gptu_multiclass_press_a_ resnet11_fit_all_gptu_multiclass_press_a_ crnn_fit_all_gptu_multiclass_press_a_ wavenet_fit_all_gptu_multiclass_press_a_

# default binary unknown/known users:
python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default_unknown_users --save_only --users unknown --basenames $bin_r $bin_r11 $bin_c $bin_w
python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default_known_users --save_only --users known --basenames $bin_r $bin_r11 $bin_c $bin_w
# default binary unknown/known data:
python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default_unknown_data --save_only --task_types unknown --basenames $bin_r $bin_r11 $bin_c $bin_w
python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default_known_data --save_only --task_types known --basenames $bin_r $bin_r11 $bin_c $bin_w

# default multiclass unknown/known users:
#python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default_unknown_users --save_only --users unknown --basenames $mul_r $mul_r11 $mul_c $mul_w
#python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default_known_users --save_only --users known --basenames $mul_r $mul_r11 $mul_c $mul_w
# default multiclass unknown/known data:
#python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default_unknown_data --save_only --task_types unknown --basenames $mul_r $mul_r11 $mul_c $mul_w
#python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default_known_data --save_only --task_types known --basenames $mul_r $mul_r11 $mul_c $mul_w

# default binary with dummy:
#python -m analysis apply_models --model_path results/bin_default --encoding binary --data_path test-data/ --save_path results/results_bin_default_dummy --save_only --basenames $bin_r $bin_r11 $bin_c $bin_w dummy
# default multiclass with dummy:
#python -m analysis apply_models --model_path results/mc_default --encoding multiclass --data_path test-data/ --save_path results/results_mc_default_dummy --save_only --basenames $mul_r $mul_r11 $mul_c $mul_w dummy
