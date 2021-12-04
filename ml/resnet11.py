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

"""
Skorch TSC ResNet-like Implementation
"""

import logging

import numpy as np
import torch
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import preprocess
from . import utils, torch_utils


LOG = logging.getLogger(__name__)

class ResidualUnitTSC(torch.nn.Module):
	"""
	ResidualUnit with 1D convolutions inspired by the work of:
	- https://arxiv.org/pdf/1512.03385.pdf
	"""
	def __init__(self, in_channels, out_channels, groups, bias=True):
		super().__init__()
		self.residual = torch.nn.Sequential(
			torch_utils.Conv1dPadSame(in_channels, out_channels, kernel_size=8, groups=groups, bias=bias),
			torch.nn.BatchNorm1d(out_channels),
			torch.nn.ReLU(),
			torch_utils.Conv1dPadSame(out_channels, out_channels, kernel_size=5, groups=groups, bias=bias),
			torch.nn.BatchNorm1d(out_channels),
			torch.nn.ReLU(),
			torch_utils.Conv1dPadSame(out_channels, out_channels, kernel_size=3, groups=groups, bias=bias),
			torch.nn.BatchNorm1d(out_channels),
		)
		if in_channels != out_channels:
			self.skip = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
		else:
			self.skip = torch.nn.Identity()
		self.output = torch.nn.ReLU()

	def forward(self, X):
		X = self.residual(X) + self.skip(X)
		return self.output(X)


class ResNet(torch.nn.Module):
	"""
	ResNet adaptation from https://arxiv.org/pdf/1611.06455.pdf and https://arxiv.org/pdf/1809.04356.pdf
	"""
	def __init__(self, channels, n_classes=1, conv_filters_pc=8, debug=False, tf_like_init=True):
		super().__init__()

		conv_filters = conv_filters_pc * channels

		self.sequence = torch.nn.Sequential(
			ResidualUnitTSC(channels, conv_filters, channels),
			ResidualUnitTSC(conv_filters, conv_filters*2, channels),
			ResidualUnitTSC(conv_filters*2, conv_filters*2, channels),
			torch.nn.AdaptiveAvgPool1d(1),
			torch.nn.Flatten(),
			torch.nn.Linear(
				in_features=conv_filters*2,
				out_features=n_classes,
			),
		)
		if tf_like_init:
			self.apply(torch_utils.init_weight_and_bias_tf_like)
		if debug:
			LOG.info(self)

	def forward(self, X):
		return self.sequence(X)


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
	batch_size: "the number of samples used in a single training pass" = 16,
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
	#callbacks.append(('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau, factor=0.5, patience=5, min_lr=0.0001)))

	# determine sampling strategy and define estimator
	train_split, iterator = torch_utils.sampling_strategy(resample_once, n_classes)
	estimator = torch_utils.get_skorch_classifier(n_classes)(
		module=ResNet,
		module__channels=len(X_labels),
		module__debug=debug,
		module__n_classes=n_classes,
		max_epochs=epochs,
		criterion=torch.nn.BCEWithLogitsLoss if n_classes == 1 else torch.nn.CrossEntropyLoss,
		criterion__weight=None if n_classes == 1 else torch.tensor(np.unique(y, return_counts=True)[1] / n_classes, dtype=torch.float32),
		batch_size=batch_size,
		optimizer=torch.optim.Adam,
		optimizer__weight_decay=0,
		lr=0.001,
		train_split=train_split(0.2),
		iterator_train=iterator,
		verbose=True,
		device="cuda" if torch.cuda.is_available() else "cpu",
		callbacks=callbacks,
	)

	# hyperparameters random grid search parameters
	param_space = {
		"batch_size": [16, 32, 64, 128],
		#"lr": [5e-3],
		# convolutional parameters
		"module__conv_filters_pc": [8, 16, 32],
		#"module__stride": [2],
		# output layer kernel regularizer
		"optimizer__weight_decay": [1e-4, 1e-5, 0], # L2 regularization
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
