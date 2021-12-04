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
Skorch WaveNet-inspired Implementation
"""

import logging

import numpy as np
import torch

import preprocess
from . import utils, torch_utils


LOG = logging.getLogger(__name__)


class GatedResidualBlock1D(torch.nn.Module):
	"""
	ResidualBlock with 1D causal dilated convolutions inspired by WaveNet.
	See the original WaveNet paper https://arxiv.org/abs/1609.03499, the ResNet paper
	https://arxiv.org/abs/1512.03385, and the gated PixelCNN paper https://arxiv.org/abs/1606.05328
	for further information.

	:param res_channels: number of output channels for the residual (usually low)
	:param kernel_size: kernel size of the dilated convolutions
	:param dilation: dilation factor, usually a power of two
	:param skip_channels: number of output channels for the skip connections (usually high)
	:param batch_norm: activate batch normalization on input
	"""
	def __init__(self, res_channels, kernel_size, dilation, skip_channels, batch_norm=True, groups=1):
		super().__init__()
		self.bn = torch.nn.BatchNorm1d(res_channels) if batch_norm else torch.nn.Identity()
		self.dilated_conv_tanh = torch.nn.Sequential(
			torch.nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0),
			torch.nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=batch_norm, groups=groups),
			torch.nn.Tanh(),
		)
		self.dilated_conv_sigmoid = torch.nn.Sequential(
			torch.nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0),
			torch.nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=batch_norm, groups=groups),
			torch.nn.Sigmoid(),
		)
		self.conv_res = torch.nn.Conv1d(res_channels, res_channels, kernel_size=1, groups=groups)
		self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, kernel_size=1, groups=groups)

	def forward(self, X):
		X = self.bn(X)
		# gated pixelCNN without conditioning, c.f. https://arxiv.org/abs/1606.05328 for formula
		gpc = self.dilated_conv_tanh(X) * self.dilated_conv_sigmoid(X)
		# c.f. Deep Voice paper https://arxiv.org/abs/1702.07825 for splitting the final convolution
		res = self.conv_res(gpc) + X
		skip = self.conv_skip(gpc)
		return res, skip


class Wavenet(torch.nn.Module):
	"""
	Inspired by the WaveNet paper: https://arxiv.org/abs/1609.03499
	NOTE: layers=2 if encoding == "binary" else 5
	"""
	def __init__(self, channels, seg_width, n_classes=1, layers=2, res_channels_pc=8, kernel_size=2,
				 skip_channels_pc=8, tf_like_init=True, debug=False):
		super().__init__()
		res_channels = res_channels_pc * channels
		skip_channels = skip_channels_pc * channels
		self.head = torch.nn.Sequential(
			torch.nn.BatchNorm1d(channels),
			torch.nn.Conv1d(channels, res_channels, kernel_size=1, bias=False, groups=channels)
		)
		self.residual_blocks = torch.nn.ModuleList([GatedResidualBlock1D(
			res_channels=res_channels,
			kernel_size=kernel_size,
			dilation=2**i,
			skip_channels=skip_channels,
			groups=channels,
		) for i in range(layers)])
		self.tail = torch.nn.Sequential(
			torch.nn.ReLU(),
			torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
			torch.nn.ReLU(),
			torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
			torch.nn.Flatten(),
			torch.nn.Linear(skip_channels * seg_width, n_classes),
			# sigmoid is incorporated in the BCEWithLogitsLoss function --> numerically more stable
			# similarly, softmax is incorporated in the CrossEntropyLoss function
		)
		if tf_like_init:
			self.apply(torch_utils.init_weight_and_bias_tf_like)
		if debug:
			LOG.info(self)

	def forward(self, X):
		# transform the number of channels to match the ones required for the residual blocks
		X = self.head(X)
		# residual blocks with skip connections
		prev_residual = X
		skip_connections = []
		for res_block in self.residual_blocks:
			prev_residual, skip = res_block(prev_residual)
			skip_connections.append(skip)
		# sum skipped connections
		X = torch.stack(skip_connections).sum(dim=0)
		# apply tail of the network
		y = self.tail(X)
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

	# determine sampling strategy and define estimator
	train_split, iterator = torch_utils.sampling_strategy(resample_once, n_classes)
	estimator = torch_utils.get_skorch_classifier(n_classes)(
		module=Wavenet,
		module__channels=len(X_labels),
		module__seg_width=seg_width,
		module__debug=debug,
		module__n_classes=n_classes,
		max_epochs=epochs,
		criterion=torch.nn.BCEWithLogitsLoss if n_classes == 1 else torch.nn.CrossEntropyLoss,
		criterion__weight=None if n_classes == 1 else torch.tensor(np.unique(y, return_counts=True)[1] / n_classes, dtype=torch.float32),
		batch_size=batch_size,
		optimizer=torch.optim.Adam,
		optimizer__weight_decay=1e-3,
		lr=5e-3, #0.001 if len(np.unique(groups)) == 1 else 0.0005,
		train_split=train_split(0.2),
		iterator_train=iterator,
		verbose=True,
		device="cuda" if torch.cuda.is_available() else "cpu",
		callbacks=callbacks,
	)

	# hyperparameters random grid search parameters
	if n_classes == 1:
		param_space = {
			# skorch hyperparameters
			#"batch_size": [128],
			"lr": [5e-2, 1e-2, 5e-3],
			# torch module parameters
			#"module__kernel_size": range(1,4),
			"module__res_channels_pc": [8, 16, 32], #32, 36, 64, 72, 108, 128, 512
			"module__skip_channels_pc": [8, 16, 32],
			"module__layers": [3, 4, 5], # 2**max conv_layers "should be" <= max seg width
			# output layer kernel regularizer
			"optimizer__weight_decay": [1e-3, 1e-4], # L2 regularization
		}
	else:
		param_space = {}

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
