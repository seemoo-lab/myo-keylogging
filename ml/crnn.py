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
Skorch CRNN Implementation
"""

import logging

import numpy as np
import torch

import preprocess
from . import utils, torch_utils


LOG = logging.getLogger(__name__)


class CRNN(torch.nn.Module):
	"""
	Inspired by the CRNN from the article DOI:10.1088/1361-6579/aacfd9 from David Burns
	"""
	def __init__(self, channels, n_classes=1, conv_layers=1, conv_filters_pc=16, kernel_size=2,
				 lstm_layers=2, lstm_units=64, lstm_dropout=0.4, lstm_bidirectional=False,
				 tf_like_init=True, debug=False):
		super().__init__()
		self.standardize = torch.nn.BatchNorm1d(channels)
		conv_filters = conv_filters_pc * channels
		self.cnn = torch.nn.Sequential(*(
			torch.nn.Sequential(
				torch_utils.Conv1dPadSame(
					in_channels=channels if i == 0 else conv_filters,
					out_channels=conv_filters,
					kernel_size=kernel_size,
					groups=channels,
					bias=False,  # disable bias if using batchnorm
				),
				torch.nn.ReLU(),
				torch.nn.BatchNorm1d(conv_filters)
			)
			for i in range(conv_layers)
		))
		self.lstm = torch.nn.LSTM(
			input_size=conv_filters,
			hidden_size=lstm_units,
			num_layers=lstm_layers,
			dropout=lstm_dropout,
			bidirectional=lstm_bidirectional
		)
		self.fc = torch.nn.Linear(
			in_features=lstm_units * lstm_layers * (2 if lstm_bidirectional else 1),
			out_features=n_classes
		)
		if tf_like_init:
			self.apply(torch_utils.init_weight_and_bias_tf_like)
		if debug:
			LOG.info(self)

	def forward(self, X):
		# standardize X
		X = self.standardize(X)
		# causal dilated convolutional layers with batch normalization
		X = self.cnn(X) # (batch x feature x seq)
		# rearrange dimensions for the stacked LSTMs with optional dropout layers in between
		X = X.permute(2, 0, 1) # (seq x batch x feature)
		o, (h, c) = self.lstm(X)
		# discard the outputs o (all hidden states of every timestep from the last LSTM layer)
		# discard the cell states c (last timestep, all LSTM layers)
		# flatten hidden states h (last timestep, all LSTM layers)
		# alternatively use h[-1] which is equivalent to o[-1]
		h = h.permute(1, 0, 2).contiguous()
		h = h.view(h.shape[0], -1)
		# feed the flattened output into the output layer - note that the application of sigmoid is
		# incoroporated into the loss function BCEWithLogitsLoss --> numerically more stable
		# similarly, softmax is incorporated in the CrossEntropyLoss function
		y = self.fc(h)
		return y

def main(
	data_path: "path to a directory to load data from" = "train-data/",
	out_path: "path to a directory to store data at" = "results/ml",
	func: "function to be executed: val, nested_hpo, shallow_hpo, fit or dry" = "fit",
	user: "the user(s) or typing style to train on (e.g. 1: user with id 1, touch_typing: all touch typists, min: the min sample, default: None: all users)" = None,
	task_types: "the task type(s) to train on (e.g. uniform, default: None: all task types)" = [],
	ignore: "specific channels or channel types to ignore" = ["quat"],
	step: "the number of samples to advance a segment" = 1,
	seg_width: "the number of timesteps within one training sample" = 50,
	target_pos: "the relative position (value between 0 and 1) of the target within a segment" = 0.6,
	n_iter: "number of iterations for the hyper-parameter optimization" = 100,
	cv: "the number of splits for the validation" = 5,
	inner_cv: "the number of splits for the nested cross validation (only for hpo)" = 3,
	epochs: "the number of epochs (full passes over all training data)" = 500,
	batch_size: "the number of samples used in a single training pass" = 128,
	lr: "the learning rate" = 0.001,
	l2: "l2 regularization" = 1e-3,
	patience: "the early stopping patience (number of epochs with no improvement after which to stop training)" = 20,
	debug: "show debug messages" = False,
	emg_pli_filter: "whether or not to filter PLI noise from raw EMG data" = False,
	emg_hp_filter: "whether or not to filter low frequencies from the raw EMG data" = False,
	encoding: "the classification to use (binary, shift, finger, multiclass_alpha or multiclass)" = "binary",
	target: "the target for encoding (state, press or release)" = "press",
	resample_once: "fix random undersampling seed for a complete training cycle" = False,
	shifted_samples: "whether to add shifted truth values to a multiclass training" = False,
	uid: "unique identifier to unambiguously distinguish runs with identical settings" = 0,
):
	# create logged function and base_name
	args = locals()
	tag = preprocess.utils.create_tag(user, task_types, encoding, target, ignore)
	logged_func, base_name = utils.log_run(getattr(utils, func), __file__, args, tag, uid, out_path)

	if debug:
		logging.getLogger().setLevel(logging.DEBUG)

	# load the specified data and segment lazily on the fly
	X, y, X_labels, n_classes, meta_data = preprocess.all_tasks_np(
		path=data_path,
		file_filter=preprocess.utils.FileFilter(user, task_types),
		ignore=ignore,
		emg_pli_filter=emg_pli_filter,
		emg_hp_filter=emg_hp_filter,
		encoding=encoding,
		target=target,
		add_shifted_samples=shifted_samples,
	)
	groups = [item["id"]["user"] for item in meta_data]
	X, y, groups = torch_utils.segment(X, y, groups, seg_width, step, target_pos)

	# define callbacks
	callbacks = torch_utils.default_callbacks(base_name, out_path, patience)

	# determine sampling strategy and define estimator
	train_split, iterator = torch_utils.sampling_strategy(resample_once, n_classes)
	estimator = torch_utils.get_skorch_classifier(n_classes)(
		module=CRNN,
		module__channels=len(X_labels),
		module__debug=debug,
		module__n_classes=n_classes,
		max_epochs=epochs,
		criterion=torch.nn.BCEWithLogitsLoss if n_classes == 1 else torch.nn.CrossEntropyLoss,
		criterion__weight=None if n_classes == 1 else torch.tensor(np.unique(y, return_counts=True)[1] / n_classes, dtype=torch.float32),
		batch_size=batch_size,
		optimizer=torch.optim.RMSprop,
		optimizer__weight_decay=l2,
		lr=lr,
		train_split=train_split(0.2),
		iterator_train=iterator,
		verbose=True,
		device="cuda" if torch.cuda.is_available() else "cpu",
		callbacks=callbacks,
	)

	# hyperparameters random grid search parameters
	param_space = {
		#"batch_size": [128],
		"lr": [1e-3, 5e-3],
		# convolutional parameters
		"module__conv_layers": [1, 2],
		#"module__conv_filters_pc": [8],
		# lstm layers
		"module__lstm_layers": [2, 4],
		"module__lstm_units": [64, 192],
		"module__lstm_dropout": [0, 0.4],
		# output layer kernel regularizer
		"optimizer__weight_decay": [1e-3, 1e-4, 1e-5], # L2 regularization
	}

	# run optimizer with logging
	logged_func(
		estimator=estimator,
		X=X,
		y=y,
		n_classes=n_classes,
		groups=groups,				# hpo and val
		n_splits_outer=cv,			# hpo and val
		n_splits_inner=inner_cv,	# hpo
		param_space=param_space,	# hpo
		n_iter=n_iter,				# hpo
	)
