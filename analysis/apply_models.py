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
Apply stored models and plot predictions versus truth.
"""

import os
import json
import copy
import pathlib

import lapsolver
import joblib
#from joblib import Memory
import pandas as pd
import numpy as np
import matplotlib
import scipy.signal
from sklearn.dummy import DummyClassifier

import preprocess
import ml.utils
import ml.torch_utils
from . import utils

#MEM = Memory("cache", verbose=0)

def create_df(samples, value=0):
	# pad with given value on the left to equalize lengths (depends on seg width and thus differs!)
	for key, entry in samples.items():
		if len(entry) < len(samples["truth"]):
			samples[key] = np.pad(samples[key],
										(len(samples["truth"]) - len(entry), 0),
										"constant",
										constant_values=value)

	# add padding of one given value at the borders for easier plotting
	for key, entry in samples.items():
		samples[key] = np.pad(samples[key], 1, "constant", constant_values=value)

	df = pd.DataFrame(samples)
	df["time [s]"] = df.index / 200
	return df


def generate_lags(y_pred, truth, distance_threshold=10):
	"""
	Generate the lags of a prediction matched to a truth within the given distance for each
	truth/prediction pair in the input. Only the first result representing the minimum absolute
	value of lags for all pairs is returned.
	Negative lags refer to a prediction matched to a later truth (i.e. the prediction comes before
	the truth), positive lags match to a prior truth (i.e. the prediction comes after the truth).
	:param y_pred: numpy array of predictions of 0 and 1
	:param truth: numpy array of the ground truth of 0 and 1
	:param distance_threshold: the allowed distance between truth and prediction to count as a pair
	:return: a numpy array of lags which minimize the total sum of the absolute value of the lags
			 and a tuple of the indices of the truth and prediction corresponding to each pair
	"""
	# find all indices corresponding to a key press
	truth_indices = np.where(truth==1)[0]
	pred_indices = np.where(y_pred==1)[0]
	# for each truth: create a list of distances for each pred index and generate a matrix
	# left of truth will be negative, right of truth positive
	matrix = np.subtract.outer(pred_indices, truth_indices).astype("float")
	# mark all predictions too far away from the truth (i.e. beyond the given distance threshold) as
	# NaN to ignore them for matching
	# this prevents false positives messing up the distances if the distance threshold is chosen
	# carefully
	matrix[np.where(abs(matrix) > distance_threshold)] = np.nan
	# get the min cost path through the matrix to find the closest predicted key press within the
	# given distance for each true key press without "reusing" key presses
	row_indices, col_indices = lapsolver.solve_dense(abs(matrix))
	#if any(abs(matrix[row_indices, col_indices]) > 20):
	#	print(truth_indices)
	#	print(pred_indices)
	#	print(truth_pred_matches)
	try:
		truth_pred_matches = list(zip(truth_indices[col_indices], pred_indices[row_indices]))
		return matrix[row_indices, col_indices], truth_pred_matches
	except IndexError:
		return None, [] # if no keystrokes were matched

def distance_cm(pred, truth, matches):
	"""
	Generate a confusion matrix which is tolerant given a temporal shift of predictions of 1
	matching a truth of 1.
	This assumes a binary classification with point-wise truth and predictions.
	It reflects the temporal uncertainty inherent in most time series classification problems.
	:param pred: numpy array of predictions of 0 and 1
	:param truth: numpy array of the ground truth of 0 and 1
	:param matches: the number of successful matches of truth and prediction within a certain
					distance in front and behind the truth
	:param distance_threshold: the allowed distance between truth and prediction to count as a match
	:return: the entries of the temporal confusion matrix, i.e. true positives, false positives,
	         false negatives, true negatives
	"""
	tn, fp, fn, tp = 0, 0, 0, 0

	tp = matches
	fp = pred.tolist().count(1) - matches
	fn = truth.tolist().count(1) - matches
	tn = pred.tolist().count(0) - fn

	return tn, fp, fn, tp # return in same format as sklearn confusion matrix

def convert_cm_to_truth_pred_pair(tn, fp, fn, tp):
	# create numpy arrays that mimick the distribution of the confusion matrix for direct use with
	# the sklearn metrics
	new_truth = np.array((tp+fn)*[1] + (tn+fp)*[0])
	new_pred = np.array(tp*[1] + fn*[0] + tn*[0] + fp*[1])
	return new_pred, new_truth

def temporal_score(y, truth, distances):
	t_scores_array = []
	for distance in distances:
		# generate lags measure and truth/prediction matches
		lags, match_indices = generate_lags(y, truth, distance)
		# create a confusion matrix and score based on its values
		t_cm = distance_cm(y, truth, len(match_indices))
		t_pred, t_truth = convert_cm_to_truth_pred_pair(*t_cm)
		t_scores, _, _, _ = ml.utils.multi_score(t_pred, None, t_truth, n_classes=1, verbose=0)
		t_scores["method"] = f"temp_{distance}"
		t_scores["lags"] = lags
		t_scores["matches"] = match_indices
		t_scores_array.append(t_scores)
#	print(f"true {truth.tolist().count(1)}")
#	print(f"pred {t_cm[2] + t_cm[3]}")
	return t_scores_array

def gather_results(samples, final_data, model, X_seg, truth, n_classes, ith_estimator, ith_estimator_pred, ith_estimator_peak, user, task_type, target, chain=None):
	distances = [1,2,3,4,5,6,7,8,9,10]
	if not X_seg:
		return samples, final_data
	# get y predictions (segmented)
	y_pred, y_prob = ml.utils.predict(model, X_seg, n_classes)

	# evaluate typing speed based on truth data
	# convert keypress samples to minutes
	_, keycode_counts = np.unique(truth, return_counts=True)
	typing_speed = sum(keycode_counts[1:])/(len(truth)*0.005)*60

	# add biased result considering a "perfect" binary prediction (i.e. all positions within the
	# temporal tolerance that match will be evaluated in multiclass, no false predictions)
	if chain is not None and n_classes > 1:
		try: # if no binary predictions were made, add no scores
			# only consider matching predictions to matching truth
			y_pred_bias = y_pred[chain[0]]
			truth_bias = truth[chain[1]]
			y_prob_bias = y_prob[chain[0]]
			b_scores, _, _, _ = ml.utils.multi_score(y_pred_bias, y_prob_bias, truth_bias, n_classes)
			final_data["lags"].append(None)
			final_data["method"].append(f"temp_bias_{chain[-1]}")
			for key in b_scores:
				final_data.setdefault(key, []).append(b_scores[key])
			final_data["classifier"].append(ith_estimator)
			final_data["participant"].append(user)
			final_data["task_type"].append(task_type)
			final_data["typing_speed"].append(typing_speed)
		except:
			pass

	# score
	if chain is not None and n_classes == 1:
		# for binary chain equals the chosen distance
		distances = chain
	if chain is not None and n_classes > 1:
		# for mc chain equals: (pred_matches, truth_matches, false_pred, false_neg)
		#print(chain)
		# rebuild prediction and truth according to matches in binary prediction to be able to score
		# in chained multiclass encoding (this knowledge is only for evaluation and not prediction)
		# tp: matches, tn: don't exist, fp: should be non-existent class -1 -> are not present in truth, fn: should be mc pred but are not present in pred
		wrong_pred_val = n_classes # -1 in comments
		try:
			# new pred: matching predictions, predictions that are not -1, -1 for all missed truth
			y_pred = np.append(np.append(y_pred[chain[0]], y_pred[chain[2]]), np.full(len(chain[3]), wrong_pred_val))
			# new pred: matching truth, -1 for all false predictions, truth for all missing predictions
			truth = np.append(np.append(truth[chain[1]], np.full(len(chain[2]), wrong_pred_val)), truth[chain[3]])
			# new prob: matching truth and 0 for class -1
			#			predictions that are not -1 and 0 for class -1
			#			0 for all classes that are not -1 and 1 for class -1
			y_prob_new = np.concatenate((y_prob[chain[0]], y_prob[chain[2]], np.full([len(chain[3]), n_classes], 0)), axis=0)
		except UnboundLocalError:
			# if no binary predictions were made only add false negative equivalents
			y_pred = np.full(len(chain[3]), wrong_pred_val)
			y_prob_new = np.full([len(chain[3]), n_classes], 0)
		pred_update = np.array([0]*y_prob_new.shape[0])
		pred_update[-len(chain[3]):] = 1
		y_prob = np.concatenate((y_prob_new, np.expand_dims(pred_update, axis=1)), axis=1)
		n_classes += 1
	scores, cm, roc_curve, pr_curve = ml.utils.multi_score(y_pred, y_prob, truth, n_classes)
	scores["lags"] = None
	scores["method"] = "end-to-end"
	# chaining always comes with a distance, update method
	if chain is not None and n_classes > 1:
		scores["method"] = f"temp_{chain[-1]}"
	scores_array = [scores]

	# peak and temporal scores (binary)
	if n_classes == 1 and not target == "state":
		# estimate a single key press sample based on given prediction peaks
		# note: height should be same as y_pred interpretation
		# distance evaluated from data (pp-interval with <1% data loss)
		peak_indices = scipy.signal.find_peaks(y_prob, height=0.5, distance=5, prominence=0.05)
		y_peak = np.zeros(y_prob.size)
		y_peak[peak_indices[0]] = 1 # peak_indices[0] contains the actual peak indices
		p_scores, _, _, _ = ml.utils.multi_score(y_peak, None, truth, n_classes, verbose=0)
		p_scores["lags"] = None
		p_scores["method"] = "peak"
		scores_array.append(p_scores)
		# evaluate with relaxed temporal metrics
		t_scores = temporal_score(y_peak, truth, distances=distances)
		scores_array.extend(t_scores)

	# store predictions and probabilities
	samples[ith_estimator_pred] = y_pred
	samples[ith_estimator] = y_prob
	if n_classes == 1 and not target == "state":
		samples[ith_estimator_peak] = y_peak
	# store scores for final plot
	final_data["classifier"].extend([ith_estimator]*len(scores_array))
	final_data["participant"].extend([user]*len(scores_array))
	final_data["task_type"].extend([task_type]*len(scores_array))
	final_data["typing_speed"].extend([typing_speed]*len(scores_array))
	for scores in scores_array:
		for key in scores:
			final_data.setdefault(key, []).append(scores[key])
	#print(final_data)

	# reshape keras dimensions (except for multiclass probabilities)
	if samples[ith_estimator_pred].ndim > 1: # predictions
		samples[ith_estimator_pred] = np.ravel(samples[ith_estimator_pred])
	if n_classes == 1:
		if samples[ith_estimator].ndim > 1: # probabilities
			samples[ith_estimator] = np.ravel(samples[ith_estimator])
			samples[ith_estimator_peak] = np.ravel(samples[ith_estimator_peak])
	else:
		# maximum prediction values (class independent)
		# and values of predictions for the true class
		max_proba, truth_proba = [], []
		for row, ti in zip(samples[ith_estimator], truth):
			max_proba.append(np.max(row))
			truth_proba.append(row[int(ti)])
		samples[f"{ith_estimator}_maximum_estimate"] = max_proba
		samples[f"{ith_estimator}_truth_estimate"] = truth_proba

		# all proba data
		proba_name_mc = f"{ith_estimator}_class_"
		a = samples[ith_estimator]
		estimator_i_probas = {f"{proba_name_mc}{key}":a[:,key] for key in range(a.shape[1])}
		samples.pop(ith_estimator)
		samples = {**samples, **estimator_i_probas}

	# choose the longest truth as truth (depends on seg width and thus differs!)
	if len(samples["truth"]) < len(truth):
		samples["truth"] = truth
	return samples, final_data

def predict_and_score(encoding, estimator_names, basenames, model_path, bin_samples_counter, zero_pred_counter, n_classes, pred_matches_mul, final_data, final_data_mul, X, y, user, task_type, target, tolerance, dummy_strategy):
	# create dummy values based on the training data to fit a dummy classifier
	dummy_y_mul = [0] * 31 + [1] * 31 + [2] * 24 + [3] * 22 + [4] * 22 + [5] * 22 + [6] * 22 + [7] * 28 + [8] * 29 + [9] * 33 + [10] * 27 + [11] * 8 + [12] * 208 + [13] * 28 + [14] * 61 + [15] * 449 + [16] * 214 + [17] * 236 + [18] * 72 + [19] * 119 + [20] * 245 + [21] * 143 + [22] * 65 + [23] * 29 + [24] * 24 + [25] * 81 + [26] * 279 + [27] * 230 + [28] * 187 + [29] * 66 + [30] * 96 + [31] * 122 + [32] * 27 + [33] * 59 + [34] * 136 + [35] * 29 + [36] * 38 + [37] * 17 + [38] * 160 + [39] * 16 + [40] * 33 + [41] * 29 + [42] * 111 + [43] * 47 + [44] * 69 + [45] * 279 + [46] * 105 + [47] * 59 + [48] * 60 + [49] * 24 + [50] * 83 + [51] * 580
	dummy_y_mul_alpha = [0] * 208 +[1] * 28 +[2] * 61 +[3] * 449 +[4] * 214 +[5] * 236 +[6] * 72 +[7] * 119 +[8] * 245 +[9] * 143 +[10] * 65 +[11] * 81 +[12] * 279 +[13] * 230 +[14] * 187 +[15] * 66 +[16] * 96 +[17] * 122 +[18] * 27 +[19] * 59 +[20] * 136 +[21] * 160 +[22] * 33 +[23] * 29 +[24] * 111 +[25] * 47 +[26] * 69 +[27] * 279 +[28] * 105 +[29] * 83 +[30] * 580
	dummy_y_bin = [0] * 98 + [1] * 2 # approximation

	# create empty dictionary for thresholded predictions, prediction probabilities
	# and truth samples
	prediction_value = {name + " (threshold 0.5)": None for name in estimator_names}
	prediction_probability = {name: None for name in estimator_names}
	samples = {**prediction_value, **prediction_probability}
	samples["truth"] = []
	samples_mul_chain = copy.deepcopy(samples)
	dummy_counter = 0
	# predict with every model
	for i, name in enumerate(basenames):
		if name == "dummy":
			# create dummy data
			classtag = encoding
			if "chain" in encoding:
				subencoding = "_".join(encoding.split("_")[1:])
				if subencoding != "":
					subencoding = f"_{subencoding}"
				classtag = f"binary" if dummy_counter == 0 else f"multiclass{subencoding}"
				dummy_counter += 1
			json_data = {
				"params": {
					"encoding": classtag,
					"seg_width": 35
				}
			}
			if classtag == "multiclass":
				dummy_y = dummy_y_mul
			if classtag == "multiclass_alpha":
				dummy_y = dummy_y_mul_alpha
			if classtag == "binary":
				dummy_y = dummy_y_bin
			model = DummyClassifier(strategy=dummy_strategy).fit(dummy_y, dummy_y)
		else:
			# load meta data and model
			candidates = list(pathlib.Path(model_path).glob(f"{name}*.joblib"))
			assert len(candidates) == 1, "Please provide an exactly matching basename for the model"
			file_name_stem = candidates[0].stem
			with open(os.path.join(model_path, f"{file_name_stem}.result.json")) as jsonfile:
				json_data = json.load(jsonfile)
			model = joblib.load(os.path.join(model_path, f"{file_name_stem}.joblib"))

		classtag = json_data["params"]["encoding"]
		ith_estimator = estimator_names[i]
		#ith_estimator = estimator_names[i] + classtag
		ith_estimator_pred = ith_estimator + " (threshold 0.5)"
		ith_estimator_peak = ith_estimator + " peak"

		# segment (with step size 1)
		seg_width = json_data["params"]["seg_width"]
		target_pos = json_data["params"].get("target_pos", 1) # fallback to using the last position for the target position
		X_seg, y_seg, _ = ml.torch_utils.segment((X,), (y,), (user,), seg_width, 1, target_pos, drop_sentinels=False)

		if encoding == "binary":
			bin_samples_counter[0] += len(np.where(y_seg == 0)[0])
			bin_samples_counter[1] += len(np.where(y_seg == 1)[0])
		if encoding == "chain":
			bin_samples_counter[0] += len(np.where(y_seg >= 0)[0])
			bin_samples_counter[1] += len(np.where(y_seg < 0)[0])
		# predict and score
		if "chain" in encoding:
			# if chain, run once with y changed to become binary and n_classes=1 and the given
			# temporal tolerance
			if classtag == "binary":
				print("Running binary model in a chain...")
				y_seg[y_seg >= 0] = 1
				y_seg[y_seg == -1] = 0
				samples, final_data = gather_results(samples, final_data, model, X_seg, y_seg, 1, ith_estimator, ith_estimator_pred, ith_estimator_peak, user, task_type, target, chain=[tolerance])

			# run again with X and y reduced to the segments predicted as ones by the
			# binary model
			if "multiclass" in classtag:
				print(f"Running multiclass model in a chain{' on peak predictions' if target != 'state' else ''}...")
				# remove sentinel values from the truth (-1)
				print(f"truth: {y_seg}")
				# check for non-alpha indices and mark them with a sentinel in case only
				# alpha keys are considered
				if classtag == "multiclass_alpha":
					y_seg = pd.Series(y_seg)
					y_seg = y_seg.replace(preprocess.encode.MULTICLASS_TO_ALPHA)
				# remove all values that are not regarded as a key press from y in case only
				# alpha keys are regarded
				non_alpha_indices = np.where(y_seg >= 0)[0] # 0 is NOT "no key" here (multiclass encoding)
				y_seg = y_seg[non_alpha_indices]
				#print(f"new truth: {y_seg}")

				# get the predicted binary samples
				# default is peak detection as we want to match one truth to one keypress
				y_pred = samples[ith_estimator_peak]
				if target == "state":
					y_pred = samples[ith_estimator_pred]
				print(f"prediction: {y_pred}")
				# report results
				total_samples = y_pred.size
				try:
					y_pred_count = np.unique(y_pred, return_counts=True)[1][1]
				except IndexError:
					y_pred_count = 0
				truth_count = len(non_alpha_indices)
				try:
					print(f"Found binary predictions for {y_pred_count} keys (of {truth_count} true keys and {total_samples} samples total).")
				except IndexError:
					print(f"Found binary predictions for 0 keys (of {truth_count} true keys and {total_samples} samples total). Continuing to next file.")
					zero_pred_counter += 1
					continue
				print(f"Number of predicted and true keys {'' if truth_count == y_pred_count else 'do not '}match, there are {y_pred_count-truth_count} more predictions than truth samples.")

				# remove all values that are not predicted to be a key press from X
				non_zero_indices = np.nonzero(y_pred)[0] # 0 is "no key" here (binary predictions)
				X_seg_sub = X_seg[non_zero_indices]
				print(f"cleaned truth indices: {non_alpha_indices}")
				print(f"cleaned prediction indices: {non_zero_indices}")

				# change matched indices of binary result to account for deleted zeroes
				# this is required for the evaluation only
				# (pred, truth)
				pred_matches = []
				truth_matches = []
				for matches in final_data["matches"]:
					for match in matches:
						pred_matches.append(np.where(non_zero_indices == match[1])[0][0])
						truth_matches.append(np.where(non_alpha_indices == match[0])[0][0])
				false_pred = [j for j, _ in enumerate(non_zero_indices) if j not in pred_matches]
				false_neg = [j for j, _ in enumerate(non_alpha_indices) if j not in truth_matches]

				# update n_classes to the amount of alpha value classes
				if classtag == "multiclass_alpha":
					n_classes = max(preprocess.encode.MULTICLASS_TO_ALPHA.values()) + 1
					print("Updated to multiclass alphabet only.")
					print("n_classes: ", n_classes)
					print("unique labels: ", dict(zip(*np.unique(y, return_counts=True))))
				samples_mul_chain, final_data_mul = gather_results(samples_mul_chain, final_data_mul, model, X_seg_sub, y_seg, n_classes, ith_estimator, ith_estimator_pred, ith_estimator_peak, user, task_type, target, chain=(pred_matches, truth_matches, false_pred, false_neg, tolerance))

				# get similarity of predictions
				if pred_matches and truth_matches:
					# predict for all matched prediction positions
					X_pred = X_seg[non_zero_indices[pred_matches]]
					y_pred_raw, y_prob_raw = ml.utils.predict(model, X_pred, n_classes)

					for i in range(len(pred_matches)):
						pred_matches_mul["tolerance"].append(tolerance)
						dist_bin = abs(non_zero_indices[pred_matches[i]] - non_alpha_indices[truth_matches[i]])
						pred_matches_mul["distance"].append(dist_bin)
						pred_matches_mul["classifier"].append(ith_estimator)
						pred_matches_mul["participant"].append(user)
						scores_pred, _, _, _ = ml.utils.multi_score([y_pred_raw[i]], [y_prob_raw[i]], [y_seg[truth_matches][i]], n_classes)
						for key in scores_pred:
							pred_matches_mul.setdefault(key, []).append(scores_pred[key])
		else:
			# else just run
			print(f"Removing sentinels...")
			non_sentinels = np.where(y_seg >= 0)
			X_seg_sub = X_seg[non_sentinels]
			y_seg_sub = y_seg[non_sentinels]
			print(f"Running {encoding} model...")
			samples, final_data = gather_results(samples, final_data, model, X_seg_sub, y_seg_sub, n_classes, ith_estimator, ith_estimator_pred, ith_estimator_peak, user, task_type, target)
		# hand the matched positions of the peak detection over to the multiclass model for scoring
		if "chain" in encoding and classtag == "binary":
			continue
		# delete the matched positions after using them for the multiclass model
		final_data.pop("matches", None)
	return samples, final_data, samples_mul_chain, final_data_mul, pred_matches_mul, zero_pred_counter, bin_samples_counter


def analyse(df, df_pred, code, metrics, target, encoding, save_path, save_only, tolerance):
	#print(df)
	print("Mean performances per method:")
	score_names = sorted(list(set(df.columns) - set(["classifier", "participant", "method", "lags", "task_type", "typing_style", "typing_style", "typing_speed"])))
	top_n_metrics = set(score_names) - set(metrics) - set(["entropy"])
	#print(score_names)
	#print(top_n_metrics)

	# results storage helper
	def log_model_performance(filename_id, score_names, sort_by=["method","classifier"]):
		with open(os.path.join(save_path,f"model_performance_{filename_id}.txt"),"w") as outfile:
			outfile.write("mean:\n")
			df.groupby(by=sort_by)[score_names].mean().to_string(outfile)
			outfile.write("\n\n")
			outfile.write("std:\n")
			df.groupby(by=sort_by)[score_names].std().to_string(outfile)

	### print and record performance ###

	print(df.groupby(by="method")[score_names].mean())
	#print(df.groupby(by="method")[score_names].std())
	if save_only:
		# store mean and std performance of classifiers over all data
		log_model_performance(code, score_names)
		# store mean and std performance of classifiers per participant
		log_model_performance(f"{code}_participants", score_names, ["method","classifier","participant"])
		# store mean and std performance of classifiers per typing style
		log_model_performance(f"{code}_per_typing", score_names, ["method","classifier", "typing_style"])
		# store mean and std performance of classifiers per task type
		log_model_performance(f"{code}_per_task", score_names, ["method","classifier", "task_type"])
		# store mean and std performance of classifiers per participant and per typing style
		log_model_performance(f"{code}_participants_per_typing", score_names, ["method","classifier", "participant", "typing_style"])
		# store mean and std performance of classifiers per participant and per task type
		log_model_performance(f"{code}_participants_per_task", score_names, ["method","classifier", "participant", "task_type"])
		with open(os.path.join(save_path,f"corr_{code}.txt"),"w") as outfile:
			df.corr().to_string(outfile)
			outfile.write("\n\n")
			df.corr(method="spearman").to_string(outfile)


	### plot performance on metrics ###

	# plot performance of classifiers over all data for each method
	if df["classifier"].nunique() > 1 or (df["participant"].nunique() == 1 and df["classifier"].nunique() == 1):
		for method in df["method"].unique():
			x = "classifier"
			y = "model performance"
			title = f"model performance per classifier ({code}, {method})"
			filename = f"model_performance_{code}_{method}"
			data = df.loc[df["method"] == method]
			# plot all standard metrics
			utils.plotf(utils.plot_metric_results_bars, data=data, x=x, y=y, hue=metrics,
						title=title, path=save_path,
						filename=filename if save_only else "")
			# plot f1 and balanced acc
			utils.plotf(utils.plot_metric_results_bars, data=data, x=x, y=y, hue=["f1", "bal_acc"],
						title=title, path=save_path,
						filename=f"{filename}_f1_bal" if save_only else "")
			# if multiclass, also plot top n metrics
			if "multiclass" in code:
				utils.plotf(utils.plot_metric_results_bars, data=data, x=x, y=y, hue=top_n_metrics,
							title=title, path=save_path,
							filename=f"{filename}_top_n" if save_only else "")
	# plot performance of classifiers per participant for each method
	if df["participant"].nunique() > 1:
		for method in df["method"].unique():
			x = "participant"
			y = "model performance"
			col = "classifier"
			title = f"model performance per user ({code}, {method})"
			filename = f"model_performance_per_user_{code}_{method}"
			data = df.loc[df["method"] == method]
			utils.plotf(utils.plot_metric_results_cat, data=data, x=x, y=y,
				hue=metrics, col=col,
				title=title, path=save_path,
				filename=filename if save_only else ""
			)
			if "multiclass" in code:
				utils.plotf(utils.plot_metric_results_cat, data=data, x=x, y=y,
					hue=top_n_metrics, col=col,
					title=title, path=save_path,
					filename=f"{filename}_top_n" if save_only else ""
				)
		# plot end-to-end result for all participants in individual plots
		for c in df["classifier"].unique():
			x = "participant"
			hue = "typing_style"
			col = "classifier"
			title=f"model performance end-to-end per user ({code}, {c})"
			filename = f"model_performance_per_user_{code}_end-to-end_{c}"
			data = df.loc[(df["method"] == "end-to-end") & (df["classifier"] == c)].copy()
			for metric in metrics:
				y = f"model performance ({metric})"
				utils.plotf(utils.plot_metric_results_bars_with_dodge,
							data=data.rename(columns={metric: y}), x=x, y=y, hue=hue,
							title=title, path=save_path,
							filename=f"{filename}_{metric}" if save_only else "")
			if "multiclass" in code:
				y = "model performance"
				utils.plotf(utils.stacked_bars,
								data=data, x=x, y=y, stacks=top_n_metrics, hue=hue,
								title=title, path=save_path,
								filename=f"{filename}_top_n" if save_only else "")
				for metric in top_n_metrics:
					y = f"model performance ({metric})"
					utils.plotf(utils.plot_metric_results_bars_with_dodge,
								data=data.rename(columns={metric: y}), x=x, y=y, hue=hue,
								title=title, path=save_path,
								filename=f"{filename}_{metric}" if save_only else "")

			# plot performance per typing style
			if code == "binary":
				x = "typing_speed"
				y = "model performance"
				title=f"model performance speed end-to-end per typing style ({code}, {c})"
				filename = f"model_performance_speed_typing_style_{code}_end-to-end_{c}"
				data = df.loc[(df["method"] == "end-to-end") & (df["classifier"] == c)].copy()
				style = "typing_style"
				hue = ["f1", "bal_acc"]
				utils.plotf(utils.plot_speed_performance_scatter, data=data, x=x, y=y,
					hue=hue, style=style,
					title=title, path=save_path, filename=f"{filename}_f1_bal_acc"
				)
				data_ptt = data.groupby(["participant", "task_type", "typing_style"]).mean().reset_index()
				utils.plotf(utils.plot_speed_performance_scatter, data=data_ptt, x=x, y=y,
					hue=hue, style=style,
					title=title, path=save_path, filename=f"{filename}_f1_bal_acc_mean"
				)
				data_pt = data.groupby(["participant", "typing_style"]).mean().reset_index()
				utils.plotf(utils.plot_speed_performance_scatter, data=data_pt, x=x, y=y,
					hue=hue, style=style,
					title=title, path=save_path, filename=f"{filename}_f1_bal_acc_mean"
				)

	# plot performance per task type
	if code == "binary":
		for c in df["classifier"].unique():
			x = "typing_speed"
			y = "model performance"
			title=f"model performance speed end-to-end per task type ({code}, {c})"
			filename = f"model_performance_speed_task_type_{code}_end-to-end_{c}"
			data = df.loc[(df["method"] == "end-to-end") & (df["classifier"] == c)].copy()
			style = "task_type"
			hue = ["f1", "bal_acc"]
			data = data.rename(columns={x: "typing speed [keys/min]"})
			x = "typing speed [keys/min]"
			utils.plotf(utils.plot_speed_performance_scatter, data=data, x=x, y=y,
				hue=hue, style=style,
				title=title, path=save_path, filename=f"{filename}_f1_bal_acc"
			)
			utils.plotf(utils.plot_speed_performance_scatter, data=data, x=x, y=y, hue=hue,
				title=title, path=save_path, filename=f"{filename}"
			)
			data_pt = data.groupby(["participant", "task_type"]).mean().reset_index()
			utils.plotf(utils.plot_speed_performance_scatter, data=data_pt, x=x, y=y,
				hue=hue, style=style,
				title=title, path=save_path, filename=f"{filename}_f1_bal_acc_mean"
			)
			utils.plotf(utils.plot_speed_performance_scatter, data=data_pt, x=x, y=y, hue=hue,
				title=title, path=save_path, filename=f"{filename}_mean"
			)

	# plot metrics with temporal tolerance
	if code == "binary" and target != "state" and encoding != "chain":
		x = "tolerance"
		y = "model performance"
		hue = ["acc", "bal_acc", "f1"]
		col = "classifier"
		# plot for temp 0 to temp 10 (if available in data)
		data = df.loc[(df["method"] != "end-to-end") & (~df["method"].isin(["temp_20", "temp_100"]))].replace({"peak": "0", "temp_": ""}, regex=True)
		data = data.rename(columns={"method": x})
		data["tolerance [s]"] = data[x].astype(float) * 0.005
		x = "tolerance [s]"
		# shift colors to keep red as color for bal acc (needs exact length of palette)
		palette = [utils.sns.color_palette("Set1")[-3]] + utils.sns.color_palette("Set1", n_colors=len(hue)-1)
		utils.plotf(utils.plot_temp_metric_rel, data=data, x=x, y=y,
			hue=hue, col=col, palette=palette,
			title=f"model performance per distance", path=save_path,
			filename="model_performance_per_distance" if save_only else ""
		)
		for c in data[col].unique():
			utils.plotf(utils.plot_temp_metric_line, data=data.loc[data[col] == c], x=x, y=y,
				hue=hue, palette=palette,
				title=f"model performance per distance ({c})", path=save_path,
				filename=f"model_performance_per_distance_{c}" if save_only else ""
			)

	# store and plot lag for peak values
	if code == "binary" and target != "state" and encoding != "chain":
		x = "participant"
		y = "lags"
		hue = "classifier"
		col = "method"
		# convert lag arrays to columns and remove all but peak method
		data = df.dropna().explode("lags")
		data["lags"] = pd.to_numeric(data["lags"])
		#print(data)
		#print(f"Mean lag:")
		#print(data.groupby(by=[hue,x,col])[y].mean())
		#print(data.groupby(by=[hue,col])[y].mean())
		if save_only:
			with open(os.path.join(save_path,f"lag.txt"),"w") as outfile:
				outfile.write("mean:\n")
				data.groupby(by=[hue,col])[y].mean().to_string(outfile)
				outfile.write("\n\n")
				outfile.write("std:\n")
				data.groupby(by=[hue,col])[y].std().to_string(outfile)
				outfile.write("\n\n")
				outfile.write("mean:\n")
				data.groupby(by=[hue,x,col])[y].mean().to_string(outfile)
				outfile.write("\n\n")
				outfile.write("std:\n")
				data.groupby(by=[hue,x,col])[y].std().to_string(outfile)
		for c in data[hue].unique():
			m = f"temp_{tolerance}"
			current_df = data.loc[(data[hue] == c) & (data[col] == m)]
			utils.plotf(utils.plot_lags_dist, data=current_df, x=y,
				path=save_path, title=f"lags for classifier {c}, method {m}",
				filename=f"lag_peak_hist_{c}_{m}" if save_only else ""
			)
		utils.plotf(utils.plot_lags_box,
			data=data.loc[data[col].isin(["temp_10", "temp_20", "temp_100"])], x=x, y=y,
			hue=hue, col=col,
			path=save_path, title=f"lag", filename=f"lag_peak" if save_only else ""
		)

	# plot matching predictions for different shifts and maximum tolerance available
	if "chain" in encoding and df_pred is not None:
		x = "distance"
		y = "model performance"
		hue = "metric"
		data = df_pred
		data_top_n = pd.melt(data, id_vars=[c for c in data if not c.startswith("top")], var_name=hue, value_name=y)
		max_tolerance = data_top_n["tolerance"].max()
		assert data_top_n["distance"].max() <= max_tolerance
		data_top_n = data_top_n.loc[data_top_n["tolerance"] == max_tolerance]
		data_top_n = data_top_n.rename(columns={x: f"distance for tolerance {max_tolerance}"})
		x = f"distance for tolerance {max_tolerance}"
		for c in data_top_n["classifier"].unique():
			utils.plotf(utils.plot_temp_metric_line,
				data=data_top_n.loc[data_top_n["classifier"] == c], x=x, y=y, hue=hue,
				path=save_path, title=f"performance for true position vs distant position ({c})",
				filename=f"distance_vs_true_top_n_{c}" if save_only else ""
			)


def plot_predictions(df, dist, save_path, save_only, filename):
	x = "time [s]"
	y = "probability"
	row = "row"
	hue = "hue"
	#df["time [s]"] = df[x] * 0.005
	#x = "time [s]"
	truth_times = df.loc[(df["hue"] == "truth") & (df["probability"] == 1)][x]
	span_size = dist * 0.005 # from sample size to ms
	spans=[[[ts-span_size, ts+span_size] for ts in truth_times]] * (len([name for name in df["hue"].unique() if "peak" not in name]) - 1)
	utils.plotf(utils.plot_truth_prediction,
		data=df, x=x, y=y, row=row, hue=row, style=hue, draw_line=0.5, spans=spans,
		title=f"predictions vs ground truth (binary)",
		path=save_path, filename=filename if save_only else ""
	)

def main(
		basenames: "list of model basenames (e.g. crnn_fit_min_group_gptu_binary_eag_2019.11.13_105905) and/or dummy (for chain encoding, dummy needs to be used twice)" = [],
		model_path: "path to a directory to load models from" = "results/ml",
		data_path: "path to a directory to load test data from" = "test-data/",
		save_path: "path to a directory to save plots at, default is the same as model_path" = "",
		users: "the user(s) or typing style to predict on (e.g. 1: user with id 1, touch_typing: all touch typists, known/unknown: only predict on known/unknown users, default: all users)" = [],
		task_types: "the task type(s) to predict on (random, pwgen, xkcd, insecure, known/unknown: only predict on known/unknown data, default: all task types)" = [],
		emg_pli_filter: "whether or not to filter PLI noise from raw EMG data" = False,
		emg_hp_filter: "whether or not to filter low frequencies from the raw EMG data" = False,
		encoding: "the classification model to use (binary, shift, finger, multiclass or chain, i.e. binary + multiclass or multiclass_alpha or chain_alpha, i.e. binary + multiclass_alpha)" = "binary",
		target: "the target for encoding (state, press or release)" = "press",
		metrics: "the metrics to plot (acc bal_acc f1 prec rcall)" = ["acc", "bal_acc", "f1", "prec", "rcall"],
		tolerance: "the temporary tolerance distance to use for reporting and plotting" = 10,
		cut_to: "the keyword to cut the password data to include only one password in each file (first, last, all or None)" = "last",
		save_only: "save the plots and do not show them" = False,
		debug: "show plots after every iteration" = False,
		dummy_strategy: "if a classifier is a dummy, it will use the given strategy for prediction, default uniform" = "uniform",
		load_results: "a path string to the files of prior runs" = None,
	):
	user_filter = users
	task_filter = task_types
	# set the matplotlib renderer to the non-interactive scalable vector graphics backend
	if save_only:
		matplotlib.use("svg")

	if save_path == "":
		save_path = model_path
	else:
		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

	if load_results is None:
		### check input ###

		if not basenames:
			print("ERROR: Argument basenames needs to be defined.")
			return
		if isinstance(basenames, str):
			basenames = [basenames]
		basenames = sorted(basenames)

		# check basenames for identical settings and different estimators
		# get sensors used
		basename_run_identifier = []
		estimator_names = []
		ignored_sensors = []
		for name in basenames:
			# do not add the dummy yet to exclude it from argument checks
			if name == "dummy":
				continue
			# get estimator name and check basenames
			name_without_date = name.split(".")[0]
			if not "_fit_" in name_without_date:
				print(f"ERROR: One of the given classifiers does not seem to be a fitted model.")
				return
			estimator_name = name_without_date.split("_fit_")[0]
			estimator_names.append(estimator_name)
			basename_run_identifier.append(name_without_date[len(estimator_name):])
			# check for sensors used
			sensor_initials = name_without_date.split("_")[-2]
			if "e" not in sensor_initials:
				ignored_sensors.append("emg")
			if "a" not in sensor_initials:
				ignored_sensors.append("acc")
			if "g" not in sensor_initials:
				ignored_sensors.append("gyro")
			if "q" not in sensor_initials:
				ignored_sensors.append("quat")

		if any([target not in name for name in basename_run_identifier]):
			print(f"ERROR: The target \"{target}\" you are specifying is not the target used for training the estimator. Aborting, as this is assumed to be unintended.")
			return
		if "chain" in encoding:
			if not any(["binary" in name for name in basename_run_identifier] + ["multiclass" in name for name in basename_run_identifier]):
				print(f"ERROR: Chain encoding requires a binary and a multiclass estimator.")
				return
		else:
			if any([encoding not in name for name in basename_run_identifier]):
				print(f"ERROR: The encoding \"{encoding}\" you are specifying is not the encoding used for training all estimator(s). Aborting, as this is assumed to be unintended. Did you want to use chain encoding instead? If you are using chain encoding and you get this error message, something else is terribly wrong.")
				return
		if "chain" in encoding:
			if len(set(basename_run_identifier)) < 2:
				print("ERROR: Chain encoding requires a binary and a multiclass estimator.")
				return
			if len(set([name.replace("binary", "multiclass") for name in basename_run_identifier])) > 1 and len(set([name.replace("binary", "multiclass_alpha") for name in basename_run_identifier])) > 1:
				print("ERROR: Chain encoding requires sets of two matching estimators, i.e. all basenames must be equal, except for the estimator name, timestamp and encoding.")
				return
		elif len(set(basename_run_identifier)) > 1:
			print("ERROR: All basenames must be equal, except for the estimator name and timestamp. If you are using chain encoding and you get this error message, something else is terribly wrong.")
			return
		if len(set(estimator_names)) < len(set(basenames)-set(["dummy"])) and "chain" not in encoding or len(set(estimator_names)) < len(basenames)//2 and "chain" not in encoding:
			print("ERROR: All estimators must be different (or all estimator sets when using chain encoding).")
			return

		# add the dummy in the correct position
		# (the position is important for the plots to show up correctly!)
		for i, name in enumerate(basenames):
			if name == "dummy":
				basename_run_identifier.insert(i, "dummy")
				estimator_names.insert(i, "dummy")


		### preprocess ###

		# rename estimators
		for i, name in enumerate(estimator_names):
			if name == "wavenet": estimator_names[i] = "WaveNet adaptation"
			if name == "crnn": estimator_names[i] = "CRNN"

		# load and preprocess the data
		if user_filter and user_filter[0] in ("known", "unknown"):
			u_filter = user_filter[0]
		else:
			u_filter = None
		if task_filter and task_filter[0] in ("known", "unknown"):
			t_filter = task_filter[0]
		else:
			t_filter = []
		X_list, y_list, _, n_classes, meta_data = preprocess.all_tasks_np(
			path=data_path,
			file_filter=preprocess.utils.FileFilter(u_filter, t_filter),
			cut=False,
			ignore=ignored_sensors,
			emg_pli_filter=emg_pli_filter,
			emg_hp_filter=emg_hp_filter,
			encoding="multiclass" if "chain" in encoding else encoding,
			target=target,
			select=cut_to
		)

		users = set()
		typists = dict()
		zero_pred_counter = 0
		bin_samples_counter = [0, 0]
		final_data = {"classifier": [], "participant": [], "task_type": [], "typing_speed": []}
		final_data_mul = copy.deepcopy(final_data)
		pred_matches_mul = {"classifier": [], "participant": [], "tolerance": [], "distance": []}
		for X, y, meta in zip(X_list, y_list, meta_data):
			#print(meta["task"]["passwords"])
			#print(y[y!=-1])
			#print(f"{preprocess.encode.translate_to_text(pd.Series(y))} (de, en keymap)")

			# get the user id and task type
			user = meta["id"]["user"]
			users.add(user)
			task_type = meta["common"]["task_type"]

			# create a dictionary of typing styles mapped to users for later processing
			typist = meta["common"]["typing_style"]
			typists[user] = typist


			### load the model and predict on the test data ###
			if "chain" in encoding:
				# chaining requires running the multiclass model with the same temporal tolerance as the
				# binary model and it always works with a peak binary detection and a temporal tolerance
				distances = [1,2,3,4,5,6,7,8,9,10]
			else:
				distances = [tolerance]
			for dist in distances:
				samples, final_data, samples_mul_chain, final_data_mul, pred_matches_mul, zero_pred_counter, bin_samples_counter = predict_and_score(encoding, estimator_names, basenames, model_path, bin_samples_counter, zero_pred_counter, n_classes, pred_matches_mul, final_data, final_data_mul, X, y, user, task_type, target, dist, dummy_strategy)


				### show sample-based results ###

				if "chain" in encoding: # user 17 creates TypeErrors with empty predictions
					continue

				# create sample-based dataframes
				df_samples = create_df(samples, value=-1 if "multiclass" in encoding else 0)
				df_mul_chain = None
				if "chain" in encoding and len(samples_mul_chain["truth"]) > 0:
					df_mul_chain = create_df(samples_mul_chain, value=-1)

				# distinguish between binary and multiclass runs (binary shows predictions and truth
				# for each password, multiclass prints truth and prediction)
				if "chain" in encoding and len(samples_mul_chain["truth"]) > 0:
					df_list = ((df_samples, "binary"), (df_mul_chain, "multiclass"))
				else:
					df_list = ((df_samples, encoding),)

				for df, code in df_list:
					### binary: plot probabilities ###
					if n_classes == 1 or code == "binary" or len(samples_mul_chain["truth"]) == 0 and "chain" in encoding:
						# prepare plot
						x = "samples"
						y_ax = "probability"
						row = "row"
						hue = "hue"
						df[x] = df["time [s]"] / 0.005

						estimator_probs = [name for name in sorted(estimator_names)]
						estimator_peaks = [f"{name} peak" for name in sorted(estimator_names)]
						df = utils.columns_to_hue(df, hue, y_ax, ["truth"] + estimator_peaks + estimator_probs)
						#df = df.sort_values(by=hue, ascending=False)
						#df = pd.melt(df, id_vars=[x], value_vars=["truth"] + estimator_probs + estimator_peaks, var_name=hue, value_name=y_ax)
						df[row] = df[hue].replace({" peak": "" for name in estimator_peaks}, regex=True)
						# plot the truth last (sort it to the back of the df)
						df1 = df[df[row] == "truth"]
						df2 = df[df[row] != "truth"]
						df = df1.append(df2, ignore_index=True)

						# filename:
						# estimator1_estimator2.bin.{encoding}_user_{USER_ID}_{COLLECTION_ID}_{TASK_ID}_tolerance
						filename_head = "_".join(" ".join(set(estimator_names)).split(" ")) + "_" + basename_run_identifier[0]
						filename_head = filename_head.replace("binary_", "")
						filename_tail = encoding + "_user_" + str(user) + "_col_" + str(meta["id"]["collection"]) + "_t" + str(meta["common"]["task_id"]) + "_d" + str(dist)
						filename = filename_head + ".bin." + filename_tail
						ending = 1 # every file has two passwords, prevent overwrite
						while os.path.exists(os.path.join(save_path,f"{filename}_{str(ending)}.svg")):
							ending += 1
						filename = filename + "_" + str(ending)
						print(filename)

						if n_classes == 1:
							if target == "state":
								df = df.drop([el for el in df.columns if "peak" in el])
								df = df[~df["hue"].str.contains("peak")]
							# pickle a specific sample for reload
							#if save_only and "binary_user_0_col_10_t1_" in filename:
							#	df.to_pickle(os.path.join(save_path,f"{filename}.pkl"))
							#plot_predictions(df, dist, save_path, save_only, filename)
					#	if debug and not save_only:
					#		utils.plt.show()

					### multiclass: print the estimates ###
					#else:
						#print("truth is:")
						#print(meta["task"]["passwords"])
						#print("truth reads:")
						#if "alpha" in encoding:
						#	de_keysyms, en_keysyms = preprocess.encode.translate_to_text(df["truth"], preprocess.encode.ALPHA_TO_KEYCODE)
						#else:
						#	de_keysyms, en_keysyms = preprocess.encode.translate_to_text(df["truth"])
						#print(de_keysyms if meta["keyboard"]["layout"] == "de" else en_keysyms)
						#for name in set(estimator_names):
						#	print(f"{name} reads (threshold at 0.5):")
						#	print(df[f"{name} (threshold 0.5)"].values)
						#	if "alpha" in encoding:
						#		de_keysyms, en_keysyms = preprocess.encode.translate_to_text(df[f"{name} (threshold 0.5)"], preprocess.encode.ALPHA_TO_KEYCODE)
						#	else:
						#		de_keysyms, en_keysyms = preprocess.encode.translate_to_text(df[f"{name} (threshold 0.5)"])
						#	print(de_keysyms.replace("\n",chr(8629)) if meta["keyboard"]["layout"] == "de" else en_keysyms.replace("\n",chr(8629)))


		### show score-based results ###

		df_pred = None
		if "chain" in encoding:
			df_pred = pd.DataFrame(pred_matches_mul)
			if save_only:
				try:
					df_pred.to_pickle(os.path.join(save_path,"pred_data.pkl"))
				except:
					pass

		# distinguish between binary and multiclass runs
		if "chain" in encoding:
			data_list = ((final_data, "binary"), (final_data_mul, "multiclass"))
		else:
			data_list = ((final_data, encoding),)

		data_index = 0
		for data, code in data_list:

			### prepare data ###

			#print(data)
			#for key, value in data.items():
			#	print(key, len(value))
			df = pd.DataFrame(data)
			df["classifier"] = df["classifier"].replace({"WaveNet adaptation": "WaveNet\nadaptation"})
			df["typing_style"] = df["participant"].replace(typists) # add typing style column

			# filter users unless done already
			if user_filter and user_filter[0] not in ("known", "unknown"):
				for user in user_filter:
					try:
						int(user)
						df = df.loc(int(df["participant"]) == int(user)).copy()
					except ValueError:
						df = df.loc(df["typing_style"] == user).copy()
			# filter task types unless done already
			if task_filter and task_filter[0] not in ("known", "unknown"):
					df = df[df["task_type"].isin(task_filter)].copy()

			# store df
			if save_only:
				df.to_pickle(os.path.join(save_path,f"{code}_{data_index}.pkl"))
				data_index += 1

			analyse(df.copy(), df_pred, code, metrics, target, encoding, save_path, save_only, tolerance)
			if "chain" in encoding:
				if code == "binary":
					df_bin = df
				else:
					df_mul = df
	else:
		# if prior results are loaded
		df_files = list(pathlib.Path(load_results).glob("*.pkl"))
		# ignore user specific results
		df_files = [path for path in df_files if path.name in ("binary_0.pkl", "multiclass_0.pkl", "multiclass_1.pkl")]
		if len(df_files) == 2:
			df_bin = pd.read_pickle(pathlib.Path(load_results).joinpath("binary_0.pkl"))
			df_mul = pd.read_pickle(pathlib.Path(load_results).joinpath("multiclass_1.pkl"))
			df_pred = None
			analyse(df_bin.copy(), df_pred, "binary", metrics, target, encoding, save_path, save_only, tolerance)
			try:
				analyse(df_mul.copy(), df_pred, "multiclass", metrics, target, encoding, save_path, save_only, tolerance)
			except ValueError:
				pass
		elif len(df_files) == 1:
			df = pd.read_pickle(df_files[0])
			analyse(df, None, df_files[0].stem.split("_")[0], metrics, target, encoding, save_path, save_only, tolerance)
		else:
			raise ValueError("No valid amount of pickled dataframes found.")
		user_pkl = None
		for f in df_files:
			if "user" in f.stem:
				df_user = pd.read_pickle(pathlib.Path(load_results).joinpath(f"{f.stem}.pkl"))
				plot_predictions(df_user, tolerance, save_path, save_only, f"{f.stem}")
				break

	# plot performance for binary vs multiclass over distance
	if "chain" in encoding:
		score_names = sorted(list(set(df_mul.columns) - set(["classifier", "participant", "method", "lags", "task_type", "typing_style", "typing_style", "typing_speed"])))
		top_n_metrics = set(score_names) - set(metrics) - set(["entropy"])
		x = "method"
		y = "model performance"
		hue = "metric"
		style = "encoding"
		# prepare data
		df_bin[style] = "binary"
		df_mul[style] = "multiclass"
		df_bin = df_bin.drop(columns=["acc", "prec", "rcall"])
		df_mul = df_mul.drop(columns=["acc", "f1", "bal_acc", "prec", "rcall", "entropy"]) # rename f1 and bal_acc with *_mul if required in the plot
		rest = list(set(df_bin.columns) - set(score_names))
		df_bin = df_bin.melt(id_vars=rest, var_name=hue, value_name=y)
		rest = list(set(df_mul.columns) - set(score_names))
		df_mul = df_mul.melt(id_vars=rest, var_name=hue, value_name=y)
		# concatenate dfs
		data = pd.concat([df_bin, df_mul], ignore_index=True)
		data = data.loc[data[x].str.startswith("temp")]
		data = data.replace({"temp_": ""}, regex=True)
		data = data.rename(columns={"method": "tolerance"})
		for c in data["classifier"].unique():
			x = "tolerance"
			data_classifier = data.loc[data["classifier"] == c]
			data_non_biased = data_classifier.loc[~data_classifier[x].str.contains("bias")]
			data_non_biased["tolerance [s]"] = data_non_biased[x].astype(float) * 0.005
			x = "tolerance [s]"
			utils.plotf(utils.plot_temp_metric_bin_mul_line,
				data=data_non_biased, x=x, y=y, hue=hue, style=style,
				path=save_path, title=f"performance for binary vs multiclass ({c})",
				filename=f"performance_bin_mul_distance_{c}" if save_only else ""
			)
			x = "tolerance"
			data_biased = data_classifier.loc[data_classifier[x].str.contains("bias")]
			data_b = data_biased.replace({"bias_": ""}, regex=True)
			data_b["tolerance [s]"] = data_b[x].astype(float) * 0.005
			x = "tolerance [s]"
			utils.plotf(utils.plot_temp_metric_bin_mul_line,
				data=data_b.loc[~data_b[hue].isin(["bal_acc_mul", "f1_mul"])], x=x, y=y,
				hue=hue, style=style,
				path=save_path, title=f"performance for binary vs multiclass - biased ({c})",
				filename=f"performance_bin_mul_distance_bias_top_n_{c}" if save_only else ""
			)
			x = "tolerance"
			data_biased = data_classifier.loc[(data_classifier[x].str.contains("bias")) | (data_classifier[hue] == "bal_acc") | (data_classifier[hue] == "f1")]
			data_b = data_biased.replace({"bias_": ""}, regex=True)
			data_b["tolerance [s]"] = data_b[x].astype(float) * 0.005
			x = "tolerance [s]"
			utils.plotf(utils.plot_temp_metric_bin_mul_line,
				data=data_b.loc[~data_b[hue].isin(["bal_acc_mul", "f1_mul"])], x=x, y=y,
				hue=hue, style=style,
				path=save_path, title=f"performance for binary vs multiclass - biased ({c})",
				filename=f"performance_bin_mul_distance_bias_{c}" if save_only else ""
			)

	if load_results is None:
		if encoding != "binary":
			print(f"{zero_pred_counter} of {len(meta)} multiclass predictions total were skipped, as no keystroke was detected.")
			if save_only:
				with open(os.path.join(save_path,f"log.txt"),"w") as outfile:
					outfile.write(f"{zero_pred_counter} of {len(meta)} multiclass predictions total were skipped, as no keystroke was detected.")
		else:
			print(f"Total binary samples: {sum(bin_samples_counter)} with {bin_samples_counter[1]} keypress samples.")
			if save_only:
				with open(os.path.join(save_path,f"log.txt"),"w") as outfile:
					outfile.write(f"Total binary samples: {sum(bin_samples_counter)} with {bin_samples_counter[1]} keypress samples.")

	if not save_only:
		utils.plt.show()
