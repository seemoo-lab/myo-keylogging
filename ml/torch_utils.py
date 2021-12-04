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

import logging
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, Subset
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.helper import SliceDataset
from skorch.dataset import Dataset, CVSplit
from skorch.callbacks import Callback, Checkpoint, EarlyStopping

LOG = logging.getLogger(__name__)

############################################# CALLBACKS ############################################

class LoadInitStateAfterTraining(Callback):
	"""
	Skorch callback for loading model params stored by a Checkpoint callback after training.
	"""
	def __init__(self, checkpoint):
		self.checkpoint = checkpoint

	def on_train_end(self, net, X=None, y=None, **kwargs):
		try:
			net.load_params(checkpoint=self.checkpoint)
		except FileNotFoundError:
			LOG.warn("Failed to load the checkpoint - files are missing!")

def default_callbacks(base_name, out_path, es_patience=20):
	"""
	Default callbacks used for all neural networks.
	"""
	checkpoint = Checkpoint(monitor="valid_loss_best", fn_prefix=base_name + ".", dirname=out_path)
	load_state = LoadInitStateAfterTraining(checkpoint)
	callbacks = [
		("es", EarlyStopping(monitor="valid_loss", threshold=0.0, patience=es_patience)),
		("cp", checkpoint),
		("ls", load_state),
	]
	return callbacks

################################ SKORCH AND PYTORCH HELPER FUNCTIONS ###############################

def get_skorch_classifier(n_classes):
	"""
	Convenience function to use the same neural network with NeuralNetBinaryClassifier or
	NeuralNetClassifier classes from skorch.
	"""
	if n_classes == 1:
		return NeuralNetBinaryClassifier
	else:
		return NeuralNetClassifier

def init_weight_and_bias_tf_like(module):
	"""
	Initialize all weights and biases from Linear, Conv1d, Conv2d and LSTMs like it is done in
	TensorFlow using uniform Xavier initialization for weights, zeros for biases and orthogonal
	initalization for hidden-hidden weights.
	"""
	default_cls = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
	if any(isinstance(module, cls) for cls in default_cls):
		LOG.info("TF like init of %s", module)
		torch.nn.init.xavier_uniform_(module.weight)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
	elif isinstance(module, torch.nn.LSTM):
		LOG.info("TF like init of %s", module)
		for i in range(module.num_layers):
			torch.nn.init.orthogonal_(getattr(module, "weight_hh_l" + str(i)))
			torch.nn.init.xavier_uniform_(getattr(module, "weight_ih_l" + str(i)))
			torch.nn.init.zeros_(getattr(module, "bias_hh_l" + str(i)))
			torch.nn.init.zeros_(getattr(module, "bias_ih_l" + str(i)))

######################################## SAMPLING STRATEGIES #######################################

class RandomUnderSampler(SubsetRandomSampler):
	"""
	Sampler to determine indices used to randomly undersample (and thereby balance) a dataset.
	Note that the indices are chosen by random.

	:param y: the target vector to be used for determining the random undersampling
	"""
	def __init__(self, y):
		# determine number of classes and class counts
		class_list, class_counts = np.unique(y, return_counts=True)
		LOG.debug("Found %s different class(es)", len(class_list))
		LOG.debug("Minority class has %s item(s)", np.min(class_counts))
		# create random indices for every class
		indices = np.empty((0,), int)
		for cls in class_list:
			cls_indices = np.where(y == cls)[0]
			choice = np.random.choice(cls_indices, np.min(class_counts), replace=False)
			indices = np.concatenate([indices, choice])
		LOG.debug("Subsampling from %s to %s item(s)", len(y), len(indices))
		super().__init__(indices)

class RandomUnderSamplerDataLoader(DataLoader):
	"""
	Standard PyTorch DataLoader using the RandomUnderSampler as a sampler, thereby resampling the
	data per epoch.
	"""
	def __init__(self, dataset, **kwargs):
		if hasattr(dataset, "dataset"):
			y = dataset.dataset.y[dataset.indices]
			kwargs["sampler"] = RandomUnderSampler(y)
		else:
			# NOTE: This is a crude workaround to be able to use this data loader for iterator_valid
			# but not when evaluating the fitted estimator, e.g. during a cross-validation
			LOG.error("Skipping random undersampling")
		super().__init__(dataset, **kwargs)

class RandomSplit(CVSplit):
	"""
	Workflow:
	1. shuffle & split (per train cycle)
	2. resample & shuffle (per epoch)
	Notes:
	- needs iterator_train for resampling
	- validation data will not be resampled
	"""
	pass

class TemporalValidationSplit:
	"""
	Workflow:
	1. temporal split (once per train cycle)
	2. resample validation data (once per train cycle)
	3. resample & shuffle training data (per epoch) by setting iterator_train=RandomUnderSamplerDataLoader
	Notes:
	- needs iterator_train for resampling training data
	- resamples validation data (e.g. used for early stopping) --> overly optimistic results (actually better than not resampling)
	- temporal order ensured, i.e. training data is always older than validation data
	"""
	def __init__(self, ratio):
		self.ratio = ratio
	def __call__(self, dataset, y=None, groups=None):
		LOG.info("Using %.1f %% of %s item(s) for validation", self.ratio * 100, len(dataset))
		split_index = int(len(dataset) * (1 - self.ratio))
		idx_train = list(range(0, split_index))
		idx_valid = list(range(split_index, len(dataset)))
		LOG.info("Number of train data: %s (%s to %s)", len(idx_train), idx_train[0], idx_train[-1])
		LOG.info("Number of valid data: %s (%s to %s)", len(idx_valid), idx_valid[0], idx_valid[-1])
		dataset_train = Subset(dataset, idx_train)
		dataset_valid = Subset(dataset, idx_valid)
		subsampled_idx_valid = list(RandomUnderSampler(dataset.y[idx_valid]))
		LOG.info("Randomly subsampled valid data from %s to %s item(s)", len(idx_valid), len(subsampled_idx_valid))
		dataset_valid = Subset(dataset_valid, subsampled_idx_valid)
		LOG.info("NOTE: train data can be subsampled during every epoch")
		return dataset_train, dataset_valid

class RandomUnderSamplerSplit:
	"""
	Workflow:
	1. resample & shuffle (per train cycle)
	2. split (per train cycle)
	Notes:
	- no iterator_train required for resampling
	- resamples validation data (e.g. used for early stopping) --> overly optimistic results (actually better than not resampling)
	- no temporal order --> training and validation data are picked at random
	"""
	def __init__(self, ratio):
		self.ratio = ratio
	def __call__(self, dataset, y=None, groups=None):
		subsampled_idx = list(RandomUnderSampler(dataset.y))
		LOG.info("Randomly subsampled from %s to %s item(s)", len(dataset), len(subsampled_idx))
		LOG.info("Using %.1f %% of %s item(s) for validation", self.ratio * 100, len(subsampled_idx))
		split_index = int(len(subsampled_idx) * (1 - self.ratio))
		idx_train = subsampled_idx[0:split_index]
		idx_valid = subsampled_idx[split_index:-1]
		LOG.info("Number of (randomly selected) train data: %s", len(idx_train))
		LOG.info("Number of (randomly selected) valid data: %s", len(idx_valid))
		dataset_train = Subset(dataset, idx_train)
		dataset_valid = Subset(dataset, idx_valid)
		return dataset_train, dataset_valid

def sampling_strategy(resample_once, n_classes):
	"""
	Choose a sampling strategy to decide how the data will be split into training and validation
	data and provide a data loader for optional resampling per epoch. Multiclass data will not be
	resampled.

	:param resample_once: resample the data before splitting it
	:param n_classes: the number of classes that may be present in the target vector
	"""
	if n_classes > 1:
		LOG.info("Subsampling disabled (for multiclass target).")
		return TemporalValidationSplit, DataLoader
	if resample_once:
		LOG.info("Subsample once before splitting.")
		return RandomUnderSamplerSplit, DataLoader
	LOG.info("Subsample once per epoch.")
	return TemporalValidationSplit, RandomUnderSamplerDataLoader

######################################### DATA SEGMENTATION ########################################

class SegmentDataset(Dataset):
	"""
	Skorch compatible dataset with lazily evaluated segmentation applied on accessing the data.

	:param X: list of input vectors
	:param seg_width: number of samples contained in a single segment
	:param step: number of samples between consecutive segments
	"""
	def __init__(self, X, seg_width, step):
		self.X = torch.as_tensor(X)
		self.seg_width = seg_width
		self.step = step

	def __len__(self):
		# the formula below is equivalent to:
		# (len(self.X) - self.seg_width)) // self.step + 1
		# given seg_width and step are positive integers
		new_len = len(self.X) - self.seg_width + 1
		return math.ceil(new_len / self.step)

	def __getitem__(self, i):
		new_i = self.step * i
		Xi = self.X[new_i:new_i + self.seg_width].transpose_(0, 1) # shape: channels x seg_width
		# return the item as a tuple to be compatible with skorch SliceDataset
		return Xi,

def segment(X, y, groups, seg_width, step, target_pos, drop_sentinels=True):
	"""
	Segment the data for each file and concatenate the results.

	:param X: list of input vectors
	:param y: list of target vectors
	:param groups: list of group vectors
	:param seg_width: number of samples contained in a single segment
	:param step: number of samples between consecutive segments
	:param target_pos: value between 0 and 1 to determine which sample belongs to the ground truth, with
	               0 being the first value, and 1 being the last value and else in between
	:param drop_sentinels: remove samples where y equals -1 (sentinels)
	:return Xt: list of transformed input vectors
	:return yt: list of transformed target vectors
	:return gt: list of transformed group vectors
	"""
	if step < 1 or step > seg_width:
		LOG.warn("step should be in [1, seg_width=%s] (was %s)", seg_width, step)
	assert target_pos <= 1 and target_pos >= 0
	Xt_list, yt_list, gt_list = [], [], []
	for sub_X, sub_y, sub_g in zip(X, y, groups):
		# transform the data
		sub_Xt = SegmentDataset(sub_X, seg_width, step)
		start_index = int(target_pos * (seg_width - 1))
		stop_index = len(sub_y) - (seg_width - 1) + start_index
		sub_yt = sub_y[start_index:stop_index:step]
		assert len(sub_Xt) == len(sub_yt)
		sub_gt = len(sub_Xt) * [sub_g]
		# drop sentinels
		if drop_sentinels:
			keep_indices = np.argwhere(sub_yt != -1).reshape(-1)
			sub_Xt = Subset(sub_Xt, keep_indices)
			sub_yt = sub_yt[keep_indices]
			sub_gt = np.array(sub_gt)[keep_indices]
			assert len(sub_Xt) == len(sub_yt) == len(sub_gt)
		# append the transformed data to individual lists
		Xt_list.append(sub_Xt)
		yt_list.append(sub_yt)
		gt_list.append(sub_gt)
	# concatenate each list and return them
	Xt = SliceDataset(ConcatDataset(Xt_list))
	yt = np.concatenate(yt_list)
	gt = np.concatenate(gt_list)
	return Xt, yt, gt

########################################## SPECIAL LAYERS ##########################################

class Conv1dPadSame(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, groups, bias):
		super().__init__()
		self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
		right_pad = kernel_size // 2
		left_pad = kernel_size - kernel_size // 2 - 1
		self.pad = torch.nn.ConstantPad1d((left_pad, right_pad), 0)

	def forward(self, X):
		return self.pad(self.conv(X))
