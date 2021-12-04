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

import sys
import logging
logging.basicConfig(format="[%(levelname)s/%(asctime)s] %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
import pprint
import datetime
import pathlib
import json
import functools
import time
import inspect
import signal

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler
import sklearn.metrics as sk_metrics
import scipy.stats as sp_stats

pp = pprint.PrettyPrinter(indent=4, width=100, compact=True)
LOG = logging.getLogger(__name__)

########################## raise a custom exception on receiving SIGUSR1 ###########################

class AbortExecution(Exception):
	pass

def handler(signum, frame):
	raise AbortExecution("Aborting after receiving SIGUSR1...")

signal.signal(signal.SIGUSR1, handler)

########################################### other utility ##########################################

class CustomEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return " ".join(str(obj).replace("\n", "").split())

def get_indices(X_labels, keys):
	assert not isinstance(keys, str)
	result = []
	for i, item in enumerate(X_labels):
		for key in keys:
			if key in item:
				result.append(i)
				break
	return result

################################## custom scorer ###################################################

def subsample_curve(score):
	subsample_len = 200
	curve = []
	for values in score:
		if len(values) > subsample_len:
			curve.append(np.append(values[::(len(values)//subsample_len)], values[-1]))
		else:
			curve.append(values)
	return curve

def top_n_accuracy_single_entry(y_proba, y_true, ns):
	"""Get the class (index) of the true value and find it in the rank of the probabilities."""
	sorted_idx = np.argsort(y_proba)
	# reverse from highest to lowest probability (because argsort has only upwards sort)
	top_correct_guess_idx = np.where(sorted_idx[::-1] == y_true)[0][0]
	# return a bitmask indicating whether the correct class was within a certain rank
	return [1 if n > top_correct_guess_idx else 0 for n in ns]

def top_n_accuracy(y_proba, y_true, ns=[3,5,10,25]):
	# calculate the top-n accuracy as average across all samples
	top_n_result = []
	for pred, truth in zip(y_proba, y_true):
		top_n_result.append(top_n_accuracy_single_entry(pred, truth, ns))
	top_n_result = np.array(top_n_result)
	return np.mean(top_n_result, axis=0)

def top_n_accuracy_alternative(y_proba, y_true, ns):
	# calculate the top-n accuracy as average across all samples
	# slower alternative to above
	top_n_results = np.zeros((len(y_true), len(ns)))
	for i in range(len(ns)):
		top_n = np.argsort(y_proba, axis=1)[:,-ns[i]:]
		for j in range(len(y_true)):
			top_n_results[j][i] = 1 if y_true[j] in top_n[j] else 0
	return np.mean(top_n_results, axis=0)

def multi_score(y_pred, y_proba, y_true, n_classes, verbose=2):
	# calculate confusion matrix and different scores
	if verbose != 0:
		LOG.info("Scoring...")
	average = "macro" if n_classes > 1 else "binary"
	score_metrics = {
		"acc":     sk_metrics.accuracy_score,
		"bal_acc": sk_metrics.balanced_accuracy_score,
		"f1":      functools.partial(sk_metrics.f1_score,        average=average),
		"prec":    functools.partial(sk_metrics.precision_score, average=average),
		"rcall":   functools.partial(sk_metrics.recall_score,    average=average),
	}
	# calculate specificity (only for binary, pos label does not work for multiclass)
	if n_classes == 1:
		score_metrics["spe"] = functools.partial(sk_metrics.recall_score, pos_label=0)
	scores = {name: metric(y_true, y_pred) for name, metric in score_metrics.items()}
	cm = sk_metrics.confusion_matrix(y_true, y_pred)

	# if multiclass calculate entropy and top-n accuracy
	if n_classes > 1:
		scores["entropy"] = sp_stats.entropy(y_pred, base=2)
		ns = [3,5,10,25] # the values for n in the top-n accuracy
		top_n_accuracies = top_n_accuracy(y_proba, y_true, ns)
		for i, n in enumerate(ns):
			scores[f"top{n}_acc"] = top_n_accuracies[i]

	# if binary retrieve ROC and PR curve
	if n_classes == 1 and y_proba is not None:
		if verbose == 2:
			LOG.info("Creating and subsampling ROC and PR curve...")
		roc_curve = subsample_curve(sk_metrics.roc_curve(y_true, y_proba))
		pr_curve = subsample_curve(sk_metrics.precision_recall_curve(y_true, y_proba))
	else:
		roc_curve, pr_curve = None, None

	# log metrics
	unique_labels = [0, 1] if n_classes == 1 else list(range(n_classes))
	if verbose == 2:
		LOG.info("balanced accuracy: %.3f, f1 score: %.3f\n%s",
			scores["bal_acc"],
			scores["f1"],
			sk_metrics.classification_report(y_true, y_pred, labels=unique_labels),
		)

	return scores, cm, roc_curve, pr_curve

def predict(estimator, X, n_classes):
	LOG.info("Predicting probabilities and labels...")
	if n_classes == 1:
		y_proba = estimator.predict_proba(X)[:, 1]
		y_pred = (y_proba > 0.5).astype("uint8")
	else:
		y_proba = estimator.predict_proba(X)
		y_pred = np.argmax(y_proba, 1)
	return y_pred, y_proba

def predict_multi_score(estimator, X, y, n_classes):
	y_pred, y_proba = predict(estimator, X, n_classes)
	return multi_score(y_pred, y_proba, y, n_classes)

def predict_single_score(estimator, X, y, n_classes, metric_name="f1"):
	y_pred, _ = predict(estimator, X, n_classes)
	if metric_name == "f1":
		score = sk_metrics.f1_score(y, y_pred, average="macro" if n_classes > 1 else "binary")
	elif metric_name == "bal_acc":
		score = sk_metrics.balanced_accuracy_score(y, y_pred)
	return score

def get_cp_nn_history(estimator):
	return {
		"last_epoch": estimator.history[-1]["epoch"],
		"valid_loss": estimator.history[-1]["valid_loss"],
		"train_loss": estimator.history[-1]["train_loss"],
		"valid_acc": estimator.history[-1]["valid_acc"],
	}

#################### main cross-validation and hyperparameter optimization loop ####################

def log_run(func, script_file, params, tag, uid, log_path):
	"""
	Create a logger for fit, val or hpo.

	:param func: function (fit, val or hpo) to be executed in a within the logging environment
	:param script_file: the name of the script file which will execute func
	:param params: dictionary of script parameters
	:param tag: descriptive string of the data filter parameters, sensor choice and encoding
	:param uid: unique identifier to unambiguously distinguish runs with identical settings
	:param log_path: path used for logging
	:return: decorated function and the base_name used for logging
	"""
	# determine base name of log file (and model, if any)
	estimator_name = pathlib.Path(script_file).stem
	date = datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")
	base_name =  "_".join([estimator_name, func.__name__, tag, date, str(uid)])

	# define decorator for the function to be executed
	def wrapper(**kwargs):
		# determine package versions
		ml_pkgs = [
			"numpy", "pandas", "scipy", "sklearn", "joblib",
			"torch", "skorch"
		]
		pkg_versions = {name: pkg.__version__ for name, pkg in sys.modules.items() if name in ml_pkgs}
		LOG.info("Package versions:\n%s", pp.pformat(pkg_versions))

		# determine file names
		pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
		json_file = pathlib.Path(log_path).joinpath(base_name + ".result.json")
		model_file = pathlib.Path(log_path).joinpath(base_name + ".joblib")
		LOG.info("JSON file:  %s", json_file)
		LOG.info("Model file: %s", model_file)

		# remove keyword arguments not required by the function to be called
		considered_kwargs, removed_kwargs = {}, {}
		for k, v in kwargs.items():
			if k in inspect.signature(func).parameters:
				considered_kwargs[k] = v
			else:
				removed_kwargs[k] = v
		LOG.info("Removed kwargs: %s", pp.pformat(list(removed_kwargs.keys())))

		# execute the function (hpo, val or fit) to get results and the run configuration
		results, run_config, model = func(**considered_kwargs)

		# store the meta data along with the results (and save the model, if any)
		json_data = {
			"info": {
				"estimator_name": estimator_name,
				"func": func.__name__,
				"tag": tag,
			},
			"pkg_versions": pkg_versions,
			"params": params,
			"run_config": run_config,
			"results": results,
		}
		json.dump(json_data, open(json_file, "w"), indent=4, cls=CustomEncoder)
		if model is not None:
			joblib.dump(model, model_file)

		LOG.info("Finished successfully!")

	# return wrapped function and base_name
	return functools.wraps(func)(wrapper), base_name

def nested_hpo(
		estimator, X, y, n_classes, groups=None, n_splits_outer=5, n_splits_inner=3, fit_params={},
		param_space={}, n_iter=10, hpo_metric="f1",
	):
	"""
	Conduct a nested hyperparameter optimization to find optimal parameter sets per fold for
	estimating the predictive performance with the val function.

	:param estimator: an estimator to be fitted (e.g. a Skorch classifier)
	:param X: input data for fitting the final estimator
	:param y: target data for fitting the final estimator
	:param n_classes: number of classes to be predicted
	:param groups: None or group labels used for splitting the data (only outer layer)
	:param n_splits_outer: number of outer validation folds (GroupKFold or KFold)
	:param n_splits_inner: number of inner validation folds (KFold)
	:param fit_params: dict passed to the fit calls of the outer validation
	:param param_space: dict of hyperparameter distributions or list of hyperparameters to try
	:param n_iter: number of hyperparameter settings sampled
	:param hpo_metric: the name of the evaluation metric (f1 or bal_acc)
	:return: a triplet of the results, the run configuration, and (if any) a fitted model
	"""
	# define the outer cross-validation split indices
	# - use GroupKFold unless only one group is given (to allow within-subject learning)
	if groups is not None and len(np.unique(groups)) >= n_splits_outer:
		outer_cv = GroupKFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y, groups=groups))
	else:
		outer_cv = KFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y))

	# define the inner cross-validation splitter (for nested hyperparameter optimization)
	# - ShuffleSplit yields no improvement in search time
	# - GroupKFold may work by now (c.f. sklearn for previous issue #7646)
	inner_cv = KFold(n_splits=n_splits_inner)

	# log run configuration
	run_config = {
		"estimator": str(estimator),
		"n_classes": n_classes,
		"n_samples": len(X),
		"outer_cv": outer_cv,
		"inner_cv": inner_cv,
		"param_space": param_space,
		"n_iter": n_iter,
		"groups": np.unique(groups),
	}
	LOG.info("Run parameters:\n%s", pp.pformat(run_config))
	combinations = 1
	for item in param_space.values():
		combinations *= len(item)
	LOG.info("Number of possible hyperparameter combinations: %s", pp.pformat(combinations))

	# sample params and define results structure
	param_subspace = iter(ParameterSampler(param_space, n_iter))
	results = {"hpo": {"metric": hpo_metric}}
	for i in range(n_splits_outer):
		results["hpo"][f"fold_{i}"] = []

	# run the hyperparameter optimization until finished or an AbortExecution is caught
	try:
		# nested hyper-parameter optimization without final refit and scoring
		for i in range(n_iter):
			# select next params and set them
			params = next(param_subspace)
			estimator.set_params(**params)
			# run outer validation loop
			for outer_fold, (outer_train_indices, _) in enumerate(outer_cv_indices):
				# slice data
				outer_X_train = X[outer_train_indices]
				outer_y_train = y[outer_train_indices]

				# run inner validation loop
				test_scores, fit_times, histories = [], [], []
				inner_cv_indices = inner_cv.split(outer_X_train, outer_y_train)
				for inner_fold, (train_indices, test_indices) in enumerate(inner_cv_indices):
					# slice data
					X_train = outer_X_train[train_indices]
					y_train = outer_y_train[train_indices]

					# fit on inner training data
					fit_time = time.time()
					estimator.fit(X_train, y_train)
					fit_time = time.time() - fit_time
					fit_times.append(fit_time)

					# score on inner validation data
					X_test = outer_X_train[test_indices]
					y_test = outer_y_train[test_indices]
					score = predict_single_score(estimator, X_test, y_test, n_classes, hpo_metric)
					test_scores.append(score)

					# add key metrics from neural network history of best (checkpointed) run if available
					if hasattr(estimator, "history"):
						histories.append(get_cp_nn_history(estimator))

					LOG.info("iter %s/%s, outer fold %s/%s, inner fold %s/%s: %s: %.3f",
						i + 1, n_iter,
						outer_fold + 1, n_splits_outer,
						inner_fold + 1, n_splits_inner,
						hpo_metric, score,
					)
				# skip refitting and scoring the best model on the outer fold
				results["hpo"][f"fold_{outer_fold}"].append({
					"mean_test_score": np.mean(test_scores),
					"std_test_score": np.std(test_scores),
					"mean_fit_time": np.mean(fit_times),
					"std_fit_time": np.std(fit_times),
					"params": params,
					"test_scores": test_scores,
					"fit_times": fit_times,
					"histories": histories,
				})
	except AbortExecution as e:
		LOG.info(e)
	return results, run_config, None

def val(estimator, X, y, n_classes, groups=None, n_splits_outer=5, fold_params=[], fit_params={}):
	"""
	Cross-validate a model, optionally setting specified parameters for each fold, e.g. to be
	determined by a nested hyperparameter optimization.

	:param estimator: an estimator to be fitted (e.g. a Skorch classifier)
	:param X: input data for fitting the final estimator
	:param y: target data for fitting the final estimator
	:param n_classes: number of classes to be predicted
	:param groups: None or group labels used for splitting the data (only outer layer)
	:param n_splits_outer: number of outer validation folds (GroupKFold or KFold)
	:param fold_params: list of dicts to set the hyperparameters for a given fold
	:param fit_params: dict passed to the fit calls of the outer validation
	:return: a triplet of the results, the run configuration, and (if any) a fitted model
	"""
	# define the outer cross-validation split indices
	# - use GroupKFold unless only one group is given (to allow within-subject learning)
	if groups is not None and len(np.unique(groups)) >= n_splits_outer:
		outer_cv = GroupKFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y, groups=groups))
	else:
		outer_cv = KFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y))

	# log run configuration
	run_config = {
		"estimator": str(estimator),
		"n_classes": n_classes,
		"n_samples": len(X),
		"outer_cv": outer_cv,
		"groups": np.unique(groups),
		"fold_params": fold_params,
	}
	LOG.info("Run configuration:\n%s", pp.pformat(run_config))

	# single threaded cross-validation (to avoid large memory consumption)
	results = {"scores": {}, "fit_time": {}, "cm": {}, "roc_curve": {}, "pr_curve": {}, "histories": {}}
	for i, (train_indices, test_indices) in enumerate(outer_cv_indices):
		if fold_params:
			estimator.set_params(**fold_params[i])
		# fit on training data
		X_train = X[train_indices]
		y_train = y[train_indices]
		fit_time = time.time()
		estimator.fit(X_train, y_train, **fit_params)
		fit_time = time.time() - fit_time
		# score on test data
		X_test = X[test_indices]
		y_test = y[test_indices]
		scores, cm, roc_curve, pr_curve = predict_multi_score(estimator, X_test, y_test, n_classes)
		iteration = "fold_" + str(i)
		results["scores"][iteration] = scores
		results["fit_time"][iteration] = fit_time
		results["cm"][iteration] = cm
		results["roc_curve"][iteration] = roc_curve
		results["pr_curve"][iteration] = pr_curve
		# add key metrics from neural network history of best (checkpointed) run if available
		if hasattr(estimator, "history"):
			results["histories"][iteration] = get_cp_nn_history(estimator)
	# calculate mean and std of the results
	results["scores_mean"] = pd.DataFrame(results["scores"]).mean(axis=1).to_dict()
	results["scores_std"] = pd.DataFrame(results["scores"]).std(axis=1).to_dict()
	return results, run_config, None

def shallow_hpo(
		estimator, X, y, n_classes, groups=None, n_splits_outer=5, fit_params={},
		param_space={}, n_iter=10, hpo_metric="f1",
	):
	"""
	Cross-validated hyperparameter search for finding optimal parameters to be evaluated with fit.

	:param estimator: an estimator to be fitted (e.g. a Skorch classifier)
	:param X: input data for fitting the final estimator
	:param y: target data for fitting the final estimator
	:param n_classes: number of classes to be predicted
	:param groups: None or group labels used for splitting the data (only outer layer)
	:param n_splits_outer: number of outer validation folds (GroupKFold or KFold)
	:param fit_params: dict passed to the fit calls
	:param param_space: dict of hyperparameter distributions or list of hyperparameters to try
	:param n_iter: number of hyperparameter settings sampled
	:param hpo_metric: the name of the evaluation metric (f1 or bal_acc)
	:return: a triplet of the results, the run configuration, and (if any) a fitted model
	"""
	# define the outer cross-validation split indices
	# - use GroupKFold unless only one group is given (to allow within-subject learning)
	if groups is not None and len(np.unique(groups)) >= n_splits_outer:
		outer_cv = GroupKFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y, groups=groups))
	else:
		outer_cv = KFold(n_splits=n_splits_outer)
		outer_cv_indices = list(outer_cv.split(X, y))

	# log run configuration
	run_config = {
		"estimator": str(estimator),
		"n_classes": n_classes,
		"n_samples": len(X),
		"outer_cv": outer_cv,
		"groups": np.unique(groups),
		"hpo": [],
	}
	LOG.info("Run configuration:\n%s", pp.pformat(run_config))
	combinations = 1
	for item in param_space.values():
		combinations *= len(item)
	LOG.info("Number of possible hyperparameter combinations: %s", pp.pformat(combinations))

	# sample params and define results structure
	param_subspace = iter(ParameterSampler(param_space, n_iter))
	results = {"hpo": []}

	# run the hyperparameter optimization until finished or an AbortExecution is caught
	try:
		# standard hyper-parameter optimization with final refit and scoring
		for i in range(n_iter):
			# select next params and set them
			params = next(param_subspace)
			estimator.set_params(**params)
			# run validation loop
			scores, fit_times, histories = [], [], []
			for fold, (train_indices, test_indices) in enumerate(outer_cv_indices):
				# slice data
				X_train = X[train_indices]
				y_train = y[train_indices]
				# fit on training data
				fit_time = time.time()
				estimator.fit(X_train, y_train)
				fit_time = time.time() - fit_time
				fit_times.append(fit_time)
				# score on validation data
				X_test = X[test_indices]
				y_test = y[test_indices]
				score = predict_single_score(estimator, X_test, y_test, n_classes, hpo_metric)
				scores.append(score)
				# add key metrics from neural network history of best (checkpointed) run if available
				if hasattr(estimator, "history"):
					histories.append(get_cp_nn_history(estimator))
				LOG.info("iter %s/%s, fold %s/%s, %s: %.3f",
					i + 1, n_iter,
					fold + 1, n_splits_outer,
					hpo_metric, score,
				)
			# skip refitting and scoring the best model on the outer fold
			results["hpo"].append({
				"mean_test_score": np.mean(scores),
				"std_test_score": np.std(scores),
				"mean_fit_time": np.mean(fit_times),
				"std_fit_time": np.std(fit_times),
				"params": params,
				"scores": scores,
				"fit_times": fit_times,
				"histories": histories,
			})
	except AbortExecution as e:
		LOG.info(e)
	return results, run_config, None

def fit(estimator, X, y, n_classes, fit_params={}):
	"""
	Train and evaluate a model on all data.

	:param estimator: an estimator to be fitted (e.g. a Skorch classifier)
	:param X: input data for fitting the final estimator
	:param y: target data for fitting the final estimator
	:param n_classes: number of classes to be predicted
	:param fit_params: dict passed to the fit calls
	:return: a triplet of the results, the run configuration, and (if any) a fitted model
	"""
	# log run configuration
	run_config = {"estimator": str(estimator), "n_classes": n_classes, "n_samples": len(X)}
	LOG.info("Run configuration:\n%s", pp.pformat(run_config))

	# fit the estimator
	fit_time = time.time()
	model = estimator.fit(X, y, **fit_params)
	fit_time = time.time() - fit_time

	# score fitted estimator (train score)
	scores, cm, roc_curve, pr_curve = predict_multi_score(estimator, X, y, n_classes)
	results = {
		"scores": scores,
		"fit_time": fit_time,
		"cm": cm,
		"roc_curve": roc_curve,
		"pr_curve": pr_curve,
	}
	# add key metrics from neural network history of best (checkpointed) run if available
	if hasattr(estimator, "history"):
		results["history"] = get_cp_nn_history(estimator)
	return results, run_config, model

def dry(estimator, X, y, n_classes):
	"""
	Dry run for for testing purposes.
	:param X: input data
	:param y: target data
	:param n_classes: number of classes to be predicted
	:return: empty results dict, the run configuration, and the (initialized) model
	"""
	# special case for skorch classifiers
	if hasattr(estimator, "initialize"):
		estimator.initialize()
		estimator.notify("on_train_end") # trigger tensorboard callbacks to close file handlers
	run_config = {
		"estimator": str(estimator),
		"n_classes": n_classes,
		"n_samples": len(X),
		"X_shape": tuple(X[0].shape),
		"y_shape": tuple(y[0].shape),
	}
	return {}, run_config, estimator
