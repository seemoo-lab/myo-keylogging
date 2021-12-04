#!/bin/bash -xe

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


####################################################################################################
# For the following analyses, ensure that the dataset is available
####################################################################################################

python -m analysis exp_clap_sync --prefix train-data/record-0.t1
# - clap_sync.pdf

python -m analysis exp_sensor_data_words --save_only --path train-data/
# - dtw_emg.pdf

python -m analysis exp_time_lag_dist --path train-data/
# - time_lag_dist.pdf

python -m analysis exp_timing --save_only --path train-data/
# - pp_intervals_tasks_train-data_zoom.pdf
# - pp_durations_tasks_train-data_zoom.pdf

####################################################################################################
# For the following analyses, ensure that the pickled data from create_predictions.sh is available
####################################################################################################

python -m analysis analyse_sensors --save_only --path results/results_bin_sensortest
# - senor_type_comparison_binary_eag.pdf
python -m analysis analyse_sensors --save_only --path results/results_mc_sensortest
# - senor_type_comparison_multiclass_eag.pdf

python -m analysis analyse_sample_prediction --save_only --path results/results_bin_default
# - sample_fit_all_gptu_press_eag_2021.bin.binary_user_0_col_10_t1_d10_1.pdf

python -m analysis analyse_speed --save_only --path_bin results/results_bin_default --path_mul results/results_mc_default --classifiers CRNN resnet
# - model_performance_speed_binary_end-to-end_CRNN_task_type.pdf
# - model_performance_speed_multiclass_end-to-end_resnet_task_type_top_3.pdf

python -m analysis analyse_participants --save_only --path results/results_bin_default --classifiers CRNN resnet11
# - model_performance_per_user_binary_end-to-end_CRNN_bal_acc.pdf
# - model_performance_per_user_binary_end-to-end_resnet11_bal_acc.pdf
python -m analysis analyse_participants --save_only --path results/results_mc_default --classifiers CRNN resnet
# - model_performance_per_user_multiclass_end-to-end_resnet_top_n.pdf

python -m analysis analyse_tolerance --save_only --path results/results_chain_default --classifiers CRNN resnet
# - model_performance_per_distance_CRNN.pdf
# - performance_bin_mul_distance_bias_resnet.pdf

####################################################################################################
# Remove analyses not used in the final paper
####################################################################################################

RESULT_FOLDER=results/analysis
rm ${RESULT_FOLDER}/dtw_acc.pdf
rm ${RESULT_FOLDER}/dtw_gyro.pdf
rm ${RESULT_FOLDER}/model_performance_per_distance_resnet.pdf
rm ${RESULT_FOLDER}/model_performance_per_user_binary_end-to-end_resnet11_bal_acc.pdf
rm ${RESULT_FOLDER}/model_performance_per_user_multiclass_end-to-end_CRNN_top_n.pdf
rm ${RESULT_FOLDER}/model_performance_speed_binary_end-to-end_resnet_task_type.pdf
rm ${RESULT_FOLDER}/model_performance_speed_multiclass_end-to-end_CRNN_task_type_top_3.pdf
rm ${RESULT_FOLDER}/performance_bin_mul_distance_bias_CRNN.pdf
rm ${RESULT_FOLDER}/senor_type_comparison_binary.pdf
rm ${RESULT_FOLDER}/senor_type_comparison_multiclass.pdf

####################################################################################################
# Rename figures to match the ones submitted for the paper
####################################################################################################

mv ${RESULT_FOLDER}/pp_intervals_tasks_{train-data,text}_zoom.pdf
mv ${RESULT_FOLDER}/pp_durations_tasks_{train-data,text}_zoom.pdf

####################################################################################################
# Recreate logs, ensure that the pickled data from create_predictions.sh is available
####################################################################################################

# main results for keystroke detection and key identification (with and without temporary tolerance)
python -m analysis apply_models --encoding binary --load_results results/results_bin_default --save_path results/results_bin_default --save_only
# - binary performance logs (table 2, table 3, table 4)
python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default --save_path results/results_mc_default --save_only
# - multiclass performance logs (table 5, table 6)
python -m analysis apply_models --encoding chain --load_results results/results_chain_default --save_path results/results_chain_default --save_only --tolerance 10
# - temporal shifts performance logs (e.g. section 3.5, section 3.7)

# sensor test results (emg, gyroscope, accelerometer)
python -m analysis apply_models --encoding binary --load_results results/results_bin_sensortest/emg --save_path results/results_bin_sensortest/emg --save_only
python -m analysis apply_models --encoding binary --load_results results/results_bin_sensortest/gyro --save_path results/results_bin_sensortest/gyro --save_only
python -m analysis apply_models --encoding binary --load_results results/results_bin_sensortest/acc --save_path results/results_bin_sensortest/acc --save_only
# - sensor tests binary performance logs (section 3.5.3)
python -m analysis apply_models --encoding multiclass --load_results results/results_mc_sensortest/emg --save_path results/results_mc_sensortest/emg --save_only
python -m analysis apply_models --encoding multiclass --load_results results/results_mc_sensortest/gyro --save_path results/results_mc_sensortest/gyro --save_only
python -m analysis apply_models --encoding multiclass --load_results results/results_mc_sensortest/acc --save_path results/results_mc_sensortest/acc --save_only
# - sensor tests multiclass performance logs (section 3.6.2)

# unknown user and unknown password comparison results
python -m analysis apply_models --encoding binary --load_results results/results_bin_default_unknown_users --save_path results/results_bin_default_unknown_users --save_only --users unknown
python -m analysis apply_models --encoding binary --load_results results/results_bin_default_known_users --save_path results/results_bin_default_known_users --save_only --users known
# - binary unknown/known users performance logs (section 3.5.4)
python -m analysis apply_models --encoding binary --load_results results/results_bin_default_unknown_data --save_path results/results_bin_default_unknown_data --save_only --task_types unknown
python -m analysis apply_models --encoding binary --load_results results/results_bin_default_known_data --save_path results/results_bin_default_known_data --save_only --task_types known
# - binary unknown/known passwords performance logs (section 3.5.4)
#python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default_unknown_users --save_path results/results_mc_default_unknown_users --save_only --users unknown
#python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default_known_users --save_path results/results_mc_default_known_users --save_only --users known
# - multiclass unknown/known users performance logs
#python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default_unknown_data --save_path results/results_mc_default_unknown_data --save_only --task_types unknown
#python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default_known_data --save_path results/results_mc_default_known_data --save_only --task_types known
# - multiclass unknown/known passwords performance logs

# dummy tests (binary, multiclass)
#python -m analysis apply_models --encoding binary --load_results results/results_bin_default_dummy --save_path results/results_bin_default_dummy --save_only
#python -m analysis apply_models --encoding multiclass --load_results results/results_mc_default_dummy --save_path results/results_mc_default_dummy --save_only

# recreate latex performance tables
python -m analysis print_all_the_things > results/latex_tables.txt
