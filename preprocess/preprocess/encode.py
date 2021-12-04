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

import pandas as pd
import numpy as np

KEYMAP_DE = {
	# number row, =/+ (US) and backspace
	10: ("1", "!"),
	11: ("2", '"'),
	12: ("3", "§"),
	13: ("4", "$"),
	14: ("5", "%"),
	15: ("6", "&"),
	16: ("7", "/"),
	17: ("8", "("),
	18: ("9", ")"),
	19: ("0", "="),
	20: ("ß", "?"),
	#21: disabled
	22: ("BackSpace", "BackSpace"),
	# upper row and return
	24: ("q", "Q"),
	25: ("w", "W"),
	26: ("e", "E"),
	27: ("r", "R"),
	28: ("t", "T"),
	29: ("z", "Z"),
	30: ("u", "U"),
	31: ("i", "I"),
	32: ("o", "O"),
	33: ("p", "P"),
	34: ("ü", "Ü"),
	35: ("+", "*"),
	36: ("Return", "Return"),
	# home row and #/' (DE)
	38: ("a", "A"),
	39: ("s", "S"),
	40: ("d", "D"),
	41: ("f", "F"),
	42: ("g", "G"),
	43: ("h", "H"),
	44: ("j", "J"),
	45: ("k", "K"),
	46: ("l", "L"),
	47: ("ö", "Ö"),
	48: ("ä", "Ä"),
	51: ("#", "'"),
	# shift_l, </> (DE), lower row and shift_r
	50: ("Shift_L", "Shift_L"),
	94: ("<", ">"),
	52: ("y", "Y"),
	53: ("x", "X"),
	54: ("c", "C"),
	55: ("v", "V"),
	56: ("b", "B"),
	57: ("n", "N"),
	58: ("m", "M"),
	59: (",", ";"),
	60: (".", ":"),
	61: ("-", "_"),
	62: ("Shift_R", "Shift_R"),
	# space
	65: ("space", "space"),
}

KEYMAP_US = {
	# number row, =/+ (US) and backspace
	10: ("1", "!"),
	11: ("2", "@"),
	12: ("3", "#"),
	13: ("4", "$"),
	14: ("5", "%"),
	15: ("6", "^"),
	16: ("7", "&"),
	17: ("8", "*"),
	18: ("9", "("),
	19: ("0", ")"),
	20: ("-", "_"),
	21: ("=", "+"),
	22: ("BackSpace", "BackSpace"),
	# upper row and return
	24: ("q", "Q"),
	25: ("w", "W"),
	26: ("e", "E"),
	27: ("r", "R"),
	28: ("t", "T"),
	29: ("y", "Y"),
	30: ("u", "U"),
	31: ("i", "I"),
	32: ("o", "O"),
	33: ("p", "P"),
	34: ("[", "{"),
	35: ("]", "}"),
	36: ("Return", "Return"),
	# home row and #/' (DE)
	38: ("a", "A"),
	39: ("s", "S"),
	40: ("d", "D"),
	41: ("f", "F"),
	42: ("g", "G"),
	43: ("h", "H"),
	44: ("j", "J"),
	45: ("k", "K"),
	46: ("l", "L"),
	47: (";", ":"),
	48: ("'", '"'),
	#51: disabled
	# shift_l, </> (DE), lower row and shift_r
	50: ("Shift_L", "Shift_L"),
	#94: disabled
	52: ("z", "Z"),
	53: ("x", "X"),
	54: ("c", "C"),
	55: ("v", "V"),
	56: ("b", "B"),
	57: ("n", "N"),
	58: ("m", "M"),
	59: (",", "<"),
	60: (".", ">"),
	61: ("/", "?"),
	62: ("Shift_R", "Shift_R"),
	# space
	65: ("space", "space"),
}

IGNORED_KEYSYMS = [
	"ISO_Level3_Shift", "Super_L", "Super_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
	"Down", "Up", "Left", "Right",
	"dead_acute", "dead_circumflex", "grave",
	"backslash",
	"Delete", "Home", "Tab"
]

KEYSYM_TO_CHAR = {
	"space": " ",
	"Return": "\n",
	"adiaeresis": "ä",
	"udiaeresis": "ü",
	"odiaeresis": "ö",
	"ssharp": "ß",
	"period": ".",
	"comma": ",",
	"numbersign": "#",
	"plus": "+",
	"minus": "-",
	"less": "<",
	"apostrophe": "'",
	"slash": "/",
	"semicolon": ";",
	"bracketleft": "[",
	"bracketright": "]",
	"equal": "=",
	"BackSpace": chr(767),
	"Shift_Release": chr(751),
	"Shift_L": chr(752),
	"Shift_R": chr(752),
	"Caps_Lock": chr(752)
}

KEYCODE_TO_MULTICLASS = {
	0: -1,
	10: 0, 11: 1, 12: 2, 13: 3, 14: 4, # left 1.
	15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 21: 11, 22: 12, # right 1.
	24: 13, 25: 14, 26: 15, 27: 16, 28: 17, # left 2.
	29: 18, 30: 19, 31: 20, 32: 21, 33: 22, 34: 23, 35: 24, 36: 25, # right 2.
	38: 26, 39: 27, 40: 28, 41: 29, # left 3. (home row)
	42: 30, 43: 31, 44: 32, 45: 33, 46: 34, 47: 35, 48: 36, 51: 37, # right 3. (home row)
	50: 38, 94: 39, 52: 40, 53: 41, 54: 42, 55: 43, 56: 44, # left 4.
	57: 45, 58: 46, 59: 47, 60: 48, 61: 49, 62: 50, # right 4.
	65: 51 # space
}

MULTICLASS_TO_KEYCODE = {v: k for k, v in KEYCODE_TO_MULTICLASS.items()}


KEYCODE_TO_ALPHA = {
	0: -1,
	10: -1, 11: -1, 12: -1, 13: -1, 14: -1, # left 1.
	15: -1, 16: -1, 17: -1, 18: -1, 19: -1, 20: -1, 21: -1, 22: 0, # right 1.
	24: 1, 25: 2, 26: 3, 27: 4, 28: 5, # left 2.
	29: 6, 30: 7, 31: 8, 32: 9, 33: 10, 34: -1, 35: -1, 36: 11, # right 2.
	38: 12, 39: 13, 40: 14, 41: 15, # left 3. (home row)
	42: 16, 43: 17, 44: 18, 45: 19, 46: 20, 47: -1, 48: -1, 51: -1, # right 3. (home row)
	50: 21, 94: -1, 52: 22, 53: 23, 54: 24, 55: 25, 56: 26, # left 4.
	57: 27, 58: 28, 59: -1, 60: -1, 61: -1, 62: 29, # right 4.
	65: 30 # space
}

ALPHA_TO_KEYCODE = {v: k for k, v in KEYCODE_TO_ALPHA.items()}


MULTICLASS_TO_ALPHA = {
	0: -1, 1: -1, 2: -1, 3: -1, 4: -1, # left 1.
	5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: 0, # right 1.
	13: 1, 14: 2, 15: 3, 16: 4, 17: 5, # left 2.
	18: 6, 19: 7, 20: 8, 21: 9, 22: 10, 23: -1, 24: -1, 25: 11, # right 2.
	26: 12, 27: 13, 28: 14, 29: 15, # left 3. (home row)
	30: 16, 31: 17, 32: 18, 33: 19, 34: 20, 35: -1, 36: -1, 37: -1, # right 3. (home row)
	38: 21, 39: -1, 40: 22, 41: 23, 42: 24, 43: 25, 44: 26, # left 4.
	45: 27, 46: 28, 47: -1, 48: -1, 49: -1, 50: 29, # right 4.
	51: 30 # space
}


def translate_to_text(keycodes, replace_rules=MULTICLASS_TO_KEYCODE):
	"""Translate a pandas series of keycodes to text."""
	keycodes = keycodes[keycodes >= 0]
	keycodes = keycodes.replace(replace_rules)
	keycodes_de = keycodes.map(lambda x: KEYMAP_DE.get(x, ("unknown",))[0])
	keycodes_de = keycodes_de.replace(KEYSYM_TO_CHAR)
	keycodes_en = keycodes.map(lambda x: KEYMAP_US.get(x, ("unknown",))[0])
	keycodes_en = keycodes_en.replace(KEYSYM_TO_CHAR)
	de_keysyms = "".join(keycodes_de.values)
	en_keysyms = "".join(keycodes_en.values)
	return de_keysyms, en_keysyms


def keysym_to_char(keysym):
	"""Return a single char for a given keysym."""
	# ignore characters not considered for learning
	if keysym in IGNORED_KEYSYMS:
		return chr(0x25a0)
	# replace considered characters not given as symbol
	if keysym in KEYSYM_TO_CHAR:
		return KEYSYM_TO_CHAR[keysym]
	return keysym

def create_zero_initialized_df(row_index, columns):
	"""Create a dataframe with the given rows and columns filled with zero values."""
	npz = np.zeros(shape=(len(row_index), len(columns)), dtype="float32")
	return pd.DataFrame(npz, index=row_index, columns=columns, dtype="float32")

def encode_as_columns(key_events, keycodes):
	"""Encode the key events in separate keycode columns for multilabel classification."""
	zero_df = create_zero_initialized_df(key_events.index, keycodes)
	key_events = key_events.join(zero_df)
	pressed_keys = set([])
	for row in key_events.itertuples():
		key = row[1]
		if key in keycodes and row[3] == "press":
			pressed_keys.add(key)
		elif row[3] == "release":
			pressed_keys.discard(key)
		for k in pressed_keys:
			key_events.at[row[0], k] = 1
	return key_events

def encode_id(key_events):
	"""
	Encode key events for multiclass key identification.
	Each press event keyword is encoded as it's respective keycode and each release
	event as 0 unless keystrokes overlap (i.e. multiple presses happen before any release)
	in which case only the last release is encoded as 0 and all others are encoded as
	the keycode of the latest keystroke not released yet.
	Unless multiple presses are due to a key held down in which case they
	are ignored.
	"""
	encoded_events = []
	key_queue = []
	for event in key_events.itertuples(False, None):
		if event[2] == "release":
			index = key_queue.index(event[0])
			key_queue.pop(index)
		elif not event[0] in key_queue:
			key_queue.append(event[0])
		if key_queue:
			encoded_events.append(key_queue[-1])
		else:
			encoded_events.append(0)
	return pd.Series(encoded_events, index=key_events.index)

def encode_single_id(key_events, target):
	"""
	Encode key events (either press or release) for key identification. Each event (either press or
	release) is encoded as it's respective keycode, the other event type as 0. Multiple presses due
	to a held down key are also marked with the respective keycode.
	:param key_events: list of key events
	:param target: either 'press' or 'release'
	"""
	neg_target = "press" if target == "release" else "release"
	key_events.loc[key_events["event"] == neg_target, "keycode"] = 0
	return pd.Series(key_events["keycode"].values, index=key_events.index)

def encode_keys_identification(key_events, keycodes, target):
	"""Encode the given raw key events for multiclass identification."""
	key_events = key_events.loc[key_events["keycode"].isin(keycodes)].copy() # filter keys
	if target == "state":
		key_events["y"] = encode_id(key_events)
	else:
		key_events["y"] = encode_single_id(key_events, target)
	return key_events

def remove_auto_press(key_events):
	"""Remove auto-press presses."""
	pressed_keys = set({})
	new_df = pd.DataFrame(columns=key_events.columns)
	 # use index to build new df (preserves index)
	for event in key_events.itertuples(True, None):
		if event[3] == "release":
			pressed_keys.discard(event[1])
			new_df.loc[event[0]] = event[1:]
		elif not event[1] in pressed_keys:
			pressed_keys.add(event[1])
			new_df.loc[event[0]] = event[1:]
	return new_df

def reindex_ffill_once(encoded_keys, index):
	"""
	Reindex a given list of encoded key events to a given index by applying a custom forward fill
	and keeping exactly one element. In contrast, Pandas reindex with forward fill and a limit of 1
	may omit an element.

	:param encoded_keys: encoded list of key data
	:param index: indices to be used for reindexing the key data
	:return: reindexed key data
	"""
	resampled_keys = create_zero_initialized_df(index, encoded_keys.columns)
	for event in encoded_keys.itertuples(True, None):
		# uncomment this if you want to include keys shorter than
		# the interval between two timestamps
		#if event[1] != 0:
		row_idx = index.searchsorted(event[0])
		if row_idx >= len(resampled_keys.index):
			break
		resampled_keys.iloc[row_idx] = event[1]
	return resampled_keys

def add_shift_for_identification(keys, max_shift=10):
	"""
	Add shifted copies of the truth samples for the key presses to a maximum shift of the given
	samples or half	the distance between the keystrokes (rounded down).
	ONLY use this for training as it alters the truth.
	Meant to be used for multiclass only (not for state multiclass, not for multilabel and not for
	sequence to sequence learning).

	:param keys: Pandas dataframe of time and encoded and resampled key events
	:param max_shift: the maximum shift applied to a keypress
	:return: Pandas dataframe of time and encoded and resampled key events with additional truth data
	"""
	new_keys = []
	last_key_index = 0
	last_key = -1
	for i, key in enumerate(keys["y"]):
		new_keys.append(key)
		if key >= 0:
			distance_to_last = i - last_key_index
			# add shifted truth before current
			left = i-min(max_shift, distance_to_last//2)
			new_keys[left:i] = [key for _ in new_keys[left:i]]
			# add shifted truth after last
			if last_key_index > 0:
				right = last_key_index+min(max_shift, distance_to_last//2)+1
				new_keys[last_key_index+1:right] = [last_key for _ in new_keys[last_key_index+1:right]]
			last_key_index = i
			last_key = key
	# add shift for last key if not done so already
	if keys["y"].iloc[i] == -1:
		distance_to_last = i - last_key_index
		right = last_key_index+min(max_shift, distance_to_last//2)+1
		new_keys[last_key_index+1:right] = [last_key for _ in new_keys[last_key_index+1:right]]
	keys["y"] = new_keys
	return keys

def resample(key_data, index, encoding, target, add_shifted_samples=False):
	"""
	Encode the given raw key events and resample with the given index.

	:param key_data: raw key events
	:param index: indices to be used for reindexing the key data
	:param encoding: key data encoding ('binary', 'finger', 'multiclass', 'multiclass_alpha' or 'shift')
	:param target: target (either 'state', 'press' events or 'release' events) to be encoded
	:param add_shifted_samples: whether to add shifted truth values to a multiclass training
	:return: encoded and resampled key data
	"""
	# encode a chosen subset of all keycodes, determined by the chosen encoding
	all_keycodes = list(set(list(KEYMAP_DE.keys()) + list(KEYMAP_US.keys())))
	if encoding in ("binary", "finger", "multiclass", "multiclass_alpha"):
		keycodes = all_keycodes
	elif encoding == "shift":
		keycodes = [50, 62] # "Shift_L", "Shift_R"
	else:
		raise ValueError("Unsupported encoding (was %s)" % encoding)

	# encode cleaned keys and drop unused columns
	cleaned_key_data = remove_auto_press(key_data)
	encoded_keys = encode_keys_identification(cleaned_key_data, keycodes, target)
	encoded_keys = encoded_keys.drop(["keycode", "keysym", "event"], axis=1)

	# reindex key data to the new index
	if target == "state":
		resampled_keys = encoded_keys.reindex(index, fill_value=0, method="ffill")
	elif target in ("press", "release"):
		resampled_keys = reindex_ffill_once(encoded_keys, index)
	else:
		raise ValueError("Unsupported target for encoding (was %s)" % target)

	# determine replacement rules for keycodes based on the encoding
	if encoding == "binary":
		replace_rule = {keycode: 1 for keycode in all_keycodes}
	elif encoding == "shift":
		replace_rule = {50: 1, 62: 1}
	elif encoding == "finger":
		replace_rule = {
			0: -1,
			10: 0, 24: 0, 38: 0, 50: 0, 94: 0, 52: 0,
			11: 1, 25: 1, 39: 1, 53: 1,
			12: 2, 26: 2, 40: 2, 54: 2,
			13: 3, 14: 3, 27: 3, 28: 3, 41: 3, 42: 3, 55: 3, 56: 3,
			15: 4, 16: 4, 29: 4, 30: 4, 43: 4, 44: 4, 57: 4, 58: 4,
			17: 5, 31: 5, 45: 5, 59: 5,
			18: 6, 32: 6, 46: 6, 60: 6,
			19: 7, 20: 7, 21: 7, 22: 7, 33: 7, 34: 7, 35: 7, 36: 7,
			47: 7, 48: 7, 51: 7, 61: 7, 62: 7,
			65: 8,
		}
	elif encoding == "multiclass_alpha":
		replace_rule = KEYCODE_TO_ALPHA
	elif encoding == "multiclass":
		replace_rule = KEYCODE_TO_MULTICLASS

	# replace keycodes, if neccessary, and determine the number of classes
	resampled_keys = resampled_keys.replace(replace_rule)
	n_classes = max(replace_rule.values()) + 1
	n_classes = 1 if n_classes == 2 else n_classes

	if encoding != "binary" and target != "state" and add_shifted_samples:
		resampled_keys = add_shift_for_identification(resampled_keys)

	target_dtype = "int64" if "multiclass" in encoding or "finger" in encoding else "float32"
	return resampled_keys.astype(target_dtype), n_classes
