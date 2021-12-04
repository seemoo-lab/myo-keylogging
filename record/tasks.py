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

"""Run user tasks to collect data."""
import argparse
import random
import time
from itertools import zip_longest
import curses
import os
import types
from functools import partial
import sys
import textwrap
import logging

import german_keymap
import english_keymap
import utils

import record

# decorator for static variables
# thanks to https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
	'''Define a decorator for static variables.'''
	def decorate(func):
		for k in kwargs:
			setattr(func, k, kwargs[k])
		return func
	return decorate

### curses functions called by tasks ###

def uniform_random_customization(_, stdscr, result, task, keymap, prompt, disappearing, delay, saved_task):
	"""
	If return or space are typed, replace them with their respective symbols such that they resemble
	the task characters to copy (so as not to confuse the user).
	Let the task disappear when the first key is hit if the flag is set to do so and stop the
	subtask automatically as soon as all characters have been typed.
	"""
	# let the task reappear if no result is left (i.e. backspace was pressed to delete the result)
	if disappearing and not result:
		task = saved_task.copy()
		stdscr.clear()
		utils.print_task(stdscr, prompt, "".join(task), "".join(result))
		return False

	# shotcut as everything else relies on result
	if not result:
		return False

	# change the appearance of enter and space to fit the task
	if result[-1] == "\n":
		result[-1] = chr(8629)
		stdscr.clear()
		utils.print_task(stdscr, prompt, "".join(task), "".join(result))
	if result[-1] == " ":
		result[-1] = chr(9251)
		stdscr.clear()
		utils.print_task(stdscr, prompt, "".join(task), "".join(result))

	# let the task disappear if required
	if disappearing and len(result) >= 1:
		for i, _ in enumerate(task):
			task[i] = " "

	# rewrite the task in any case because echo is deactivated
	stdscr.clear()
	utils.print_task(stdscr, prompt, "".join(task), "".join(result))

	# give the user a glimpse of the character typed last
	# should give feedback without disturbing the flow (hopefully, otherwise adjust the value)
	if len(result) >= len(task):
		stdscr.refresh()
		time.sleep(delay) # may also use curses.napms(int(delay*1000)) to not need time import
		return True
	return False

@static_vars(last_key=None)
def done_on_enter_customization(key, stdscr, result, task, keymap, prompt, *_):
	"""Leave the subtask when pressing enter."""
	if key == 10:
		stdscr.addstr("\n\n " + "\n ".join(textwrap.wrap(keymap.continue_or_stay_prompt, stdscr.getmaxyx()[1]-3)), curses.A_ITALIC)
		result.pop() # remove additional space which is printed when return is pressed
	else:
		stdscr.clear()
	utils.print_task(stdscr, prompt, task, result)
	if key == 10 and done_on_enter_customization.last_key == 10:
		return True
	done_on_enter_customization.last_key = key
	return False

def show_german_game_help(stdscr, num_missiles, num_ships):
	from alien_shower import addstr_format
	curses.use_default_colors() # use colors as used per default in the terminal
	stdscr.nodelay(True) # don't wait for input (while showing a black input screen)
	stdscr.clear()
	while stdscr.getch() != 10:
		addstr_format(stdscr, 0, 0, "Willkommen zu Alien Shower.", 2, 3)
		stdscr.addstr(2, 0, "Deine Aufgabe ist es, die Erde vor Aliens zu schützen.")
		stdscr.addstr(3, 0, "Um diese Aufgabe zu erfüllen, steht dir eine Schiffsflotte zur Verfügung.")
		stdscr.addstr(4, 0, "Du wirst alle Schiffe benötigen, um erfolgreich zu sein.")
		stdscr.addstr(6, 0, "(Drücke die EINGABETASTE, um fortzufahren...)", curses.A_ITALIC)
	stdscr.clear()
	while stdscr.getch() != 10:
		stdscr.addstr(0, 0, "Steuerung:", curses.A_UNDERLINE)
		addstr_format(stdscr, 2, 0, "Du kannst ein Schiff aktivieren, indem du die entsprechende Zahl drückst.", 3, 4, 9)
		addstr_format(stdscr, 3, 0, "Du kannst ein Schiff nach links/rechts steuern, indem du \"a\"/\"d\" drückst.", 6, 9)
		addstr_format(stdscr, 4, 0, "Du kannst schießen, indem du \"s\" drückst.", 2, 5)
		addstr_format(stdscr, 5, 0, "Du kannst die Geschwindigkeit erhöhen/senken, indem du \"+\"/\"-\" drückst.", 3, 7)
		addstr_format(stdscr, 6, 0, "Du kannst das Spiel beenden und die Wertung sehen, wenn du \"escape\" drückst.", 4, 11)
		addstr_format(stdscr, 7, 0, "Nach jeder Runde kannst du eine neue starten, indem du \"return\" drückst.", 7, 10)
		stdscr.addstr(9, 0, "(Drücke die EINGABETASTE, um fortzufahren...)", curses.A_ITALIC)
	stdscr.clear()
	while stdscr.getch() != 10:
		stdscr.addstr(0, 0, "Warnung:", curses.A_UNDERLINE)
		addstr_format(stdscr, 2, 0, f"Jedes Schiff kann sich nur {(num_ships*num_missiles)//2} mal bewegen.", 5, 6, 7)
		addstr_format(stdscr, 3, 0, f"Jedes Schiff kann nur {num_missiles} mal feuern.", 4, 5, 6)
		addstr_format(stdscr, 4, 0, "Du kannst nur ein Schiff aktivieren, das du steuerst, bis es zerstört ist.", 3, 4, 5)
		addstr_format(stdscr, 5, 0, "Du kannst kein Schiff deaktivieren.", 2, 3, 4)
		addstr_format(stdscr, 6, 0, "Du hast nur einen kleinen Zeitraum, um dich für deine nächste Aktion zu", 4, 5)
		stdscr.addstr(7, 0, "entscheiden, bevor sich die Aliens bewegen.")
		stdscr.addstr(9, 0, "(Drücke die EINGABETASTE, um fortzufahren...)", curses.A_ITALIC)
	stdscr.clear()
	while stdscr.getch() != 10:
		stdscr.addstr(0, 0, "Du gewinnst, falls:", curses.A_UNDERLINE)
		addstr_format(stdscr, 2, 0, "Du alle Aliens zerstörst.", 1, 2)
		stdscr.addstr(4, 0, "Du verlierst, falls:", curses.A_UNDERLINE)
		addstr_format(stdscr, 6, 0, "Du einen Schuss verfehlst.", 3)
		addstr_format(stdscr, 7, 0, "Die Lebenszeit eines Schiffs abläuft, ehe es seinen letzten Schuss abgibt.", 1, 4)
		addstr_format(stdscr, 8, 0, "Die Aliens den Boden berühren.", 1, 3)
		stdscr.addstr(10, 0, "(Drücke die EINGABETASTE, um fortzufahren...)", curses.A_ITALIC)
	stdscr.clear()
	while stdscr.getch() != 10:
		stdscr.addstr(0, 0, "Denke daran:", curses.A_UNDERLINE)
		addstr_format(stdscr, 2, 0, f"Aktiviere: {' '.join(list(map(lambda x: str(x % 10), range(1, num_ships + 1))))}", 0)
		addstr_format(stdscr, 3, 0, "Bewege: a d", 0)
		addstr_format(stdscr, 4, 0, "Schieße: s", 0)
		addstr_format(stdscr, 6, 0, "Regle die Geschwindigkeit: + -", 0, 1, 2)
		stdscr.addstr(8, 0, "Achte darauf, wo der nächste Alien erscheint, und plane deinen Zug.")
		stdscr.addstr(9, 0, "Aber nicht zu lange.")
		stdscr.addstr(10, 0, "Viel Spaß!")
		stdscr.addstr(12, 0, "(Drücke jetzt die EINGABETASTE, um zum Spielfeld fortzufahren, und drücke sie")
		stdscr.addstr(13, 0, "erneut, um zu beginnen...)", curses.A_ITALIC)


### task type functions ###

def gather_mistakes(mistakes, task, result, num_keys):
	"""Check for mistakes in the given result and gather them in the given mistakes array."""
	for i, (expected, given) in enumerate(zip_longest(task, result, fillvalue="")):
		if expected != given and not given == "":
			if given == chr(8629):
				given = "\\n"
			if given == chr(9251):
				given = " "
			mistakes[num_keys + i] = {"expected": expected, "given": given}

def uniform_random_task(keymap, tasks, delay, disappearing, word_length, sentence_length=1, blind=False):
	"""
	Generate a task consisting of several subtasks to record a uniformly distributed random
	keystroke dataset and record the mistakes made by the user.
	The word_length determines the number of characters in one word of a sentence and the
	sentence_length determined the number of words in the subtask, e.g. "dwk" for a word_length of
	3 and a sentence_length of 1 or "jker asdi" for a word_length of 4 and a sentence_length of 2.
	If disappearing is true, the characters to copy will disappear once the user starts typing and
	need to be remembered by the user.
	"""
	stop_all = False
	try:
		# start screen (get one enter)
		mistakes = {}
		if not blind:
			discontinue, _, num_keys = curses.wrapper(utils.wait_screen,
				keymap.start_uniform_random_memory_task_prompt if disappearing
					else keymap.start_uniform_random_task_prompt,
				True
			)
			if discontinue:
				return mistakes, ("uniform disappearing" if disappearing else "uniform") + f" {word_length}", stop_all

		# run task
		# split the task list into sentences of words given by word_length and sentence_length
		# the last subtask may have a smaller word and/or sentence size than the ones before
		stop = False
		uniform_run_subtask_partial = partial(utils.run_subtask, keymap=keymap, prompt=keymap.copy_prompt)
		custom_check_partial = partial(uniform_random_customization, disappearing=disappearing, delay=delay)
		while not len(tasks) < word_length*sentence_length and not stop:
			task = [[tasks.pop() for i in range(word_length)] for j in range(sentence_length)]
			task = chr(9251).join("".join(w) for w in task)
			if not blind:
				uniform_run_subtask = partial(uniform_run_subtask_partial,
					custom_check_func=partial(custom_check_partial, saved_task=list(task))
				)
				stop, result, add_to_num_keys = curses.wrapper(uniform_run_subtask, task=list(task))
				gather_mistakes(mistakes, list(task), result, num_keys)
				num_keys += add_to_num_keys
		# create the last subtask from the remaining characters
		if not stop and tasks:
			task = [tasks[i:i+word_length] for i in range(0, len(tasks), word_length)]
			task = chr(9251).join("".join(w) for w in task)
			if not blind:
				uniform_run_subtask = partial(uniform_run_subtask_partial,
					custom_check_func=partial(custom_check_partial, saved_task=list(task))
				)
				_, result, add_to_num_keys = curses.wrapper(uniform_run_subtask, task=list(task))
				gather_mistakes(mistakes, list(task), result, num_keys)
				num_keys += add_to_num_keys
	except (KeyboardInterrupt, SystemExit):
		stop_all = True
	return mistakes, ("uniform disappearing" if disappearing else "uniform") + f" {word_length}", stop_all

def prepare_uniform_random_task(keymap, hand, copies=1):
	"""
	Prepare the task with the given number of key-set copies. Include one random shifted character
	and shuffle the characters of each set for a randomized appearance in the task.
	The keymap determines the language of the task and the presumed keyboard layout.
	If only one hand is given only the respective characters typed with this hand are included
	(except for space and enter, which are always included).
	"""
	task = []
	for i in range(copies):
		# choose one random key per hand and use it's shift-modified version
		# all else are none-shifted
		left_index = random.randint(0, len(keymap.left_keys)-1)
		right_index = random.randint(0, len(keymap.right_keys)-1)

		# get the characters to type from the list of keys (one shifted version per hand)
		chars_left = [k[1] if i == left_index else k[0] for i, k in enumerate(keymap.left_keys)]
		chars_right = [k[1] if i == right_index else k[0] for i, k in enumerate(keymap.right_keys)]
		if hand == "left":
			copy = chars_left
		elif hand == "right":
			copy = chars_right
		else:
			copy = chars_left + chars_right
		# add enter and space
		if i == 0: # got one enter at start already
			copy.append(chr(9251))
		else:
			copy.append(chr(9251))
			copy.append(chr(8629))

		# shuffle the characters
		random.shuffle(copy)
		for char in copy:
			task.append(char)
	return task

def copy_text_task(keymap, texts, start_index, end_index):
	"""
	Generate a task consisting of several subtasks to record a text dataset and record the mistakes
	made by the user.
	Start and end index determine which slice to use from the given texts.
	The keymap determines the language of the task and the presumed keyboard layout.
	"""
	stop_all = False
	try:
		# start screen
		mistakes = {}
		discontinue, _, num_keys = curses.wrapper(utils.wait_screen, keymap.start_copy_text_prompt, True)
		if discontinue:
			return mistakes, "text", stop_all
		# run task
		start_index_fun = sum([1 if len(paragraphs) >= 2 or j % 3 == 0 else 0
			for j, paragraphs in enumerate(texts["minimal"][:start_index])])
		text_run_subtask = partial(utils.run_subtask,
			keymap=keymap,
			prompt=keymap.copy_input_prompt,
			custom_check_func=done_on_enter_customization
		)
		for i in range(start_index, end_index):
			part = texts["minimal"][i]
			for paragraph in part:
				stop, copied_text, add_to_num_keys = curses.wrapper(text_run_subtask, task=paragraph)
				gather_mistakes(mistakes, paragraph, copied_text, num_keys)
				num_keys += add_to_num_keys
				done_on_enter_customization.last_key = None
				if stop:
					return mistakes, "text", stop_all
			if len(part) >= 2 or i % 3 == 0:
				fun_part = texts["fun_facts"][start_index_fun]
				stop, copied_text, add_to_num_keys = curses.wrapper(text_run_subtask, task=fun_part[0])
				gather_mistakes(mistakes, fun_part[0], copied_text, num_keys)
				num_keys += add_to_num_keys
				start_index_fun += 1
				done_on_enter_customization.last_key = None
				if stop:
					break
	except (KeyboardInterrupt, SystemExit):
		stop_all = True
	return mistakes, "text", stop_all

def copy_pangrams_task(keymap, texts, start_index, end_index):
	"""
	Generate a task to record a pangram dataset and record the mistakes made by the user.
	Start and end index determine which slice of pangrams to use.
	The keymap determines the language of the task and the presumed keyboard layout.
	"""
	stop_all = False
	try:
		# start screen
		mistakes = {}
		discontinue, _, num_keys = curses.wrapper(utils.wait_screen, keymap.start_copy_pangram_prompt, True)
		if discontinue:
			return mistakes, "pangram", stop_all
		# run task
		for i in range(start_index, end_index):
			part = texts["pangrams"][i]
			stop, copied_text, add_to_num_keys = curses.wrapper(
				utils.run_subtask,
				keymap,
				keymap.copy_input_prompt,
				part[0],
				done_on_enter_customization
			)
			gather_mistakes(mistakes, part[0], copied_text, num_keys)
			num_keys += add_to_num_keys
			done_on_enter_customization.last_key = None
			if stop:
				break
	except (KeyboardInterrupt, SystemExit):
		stop_all = True
	return mistakes, "pangram", stop_all


### task definition functions ###

def task_1(keymap, hand, delay, blind=False):
	"""Uniform random task 1 with disappearing characters (4 as the users are not as tired now)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 4, blind=blind)

def task_2(keymap, texts, blind=False):
	"""Text copy task 1"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, 0, len(texts["minimal"])//6)

def task_3(keymap, hand, delay, blind=False):
	"""Uniform random task 2 (2)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 2, blind=blind)

def task_4(keymap, texts, blind=False):
	"""Pangram copy task 1"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_pangrams_task(keymap, texts, 0, 2)

def task_5(keymap, hand, delay, blind=False):
	"""Uniform random task 3 with disappearing characters (4)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 4, blind=blind)

def task_6(keymap, texts, blind=False):
	"""Text copy task 2"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, len(texts["minimal"])//6, (len(texts["minimal"])//6)*2)

def task_7(keymap, hand, delay, blind=False):
	"""Uniform random task 4 (2)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 2, blind=blind)

def task_8(keymap, texts, blind=False):
	"""Pangram copy task 2"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_pangrams_task(keymap, texts, 2, 4)

def task_9(keymap, hand, delay, blind=False):
	"""Uniform random task 5 with disappearing characters (3)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 3, blind=blind)

def task_10(keymap, texts, blind=False):
	"""Text copy task 3"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, (len(texts["minimal"])//6)*2, (len(texts["minimal"])//6)*3)

def task_11(keymap, hand, delay, blind=False):
	"""Uniform random task 6 (5)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 5, blind=blind)

def task_12(keymap, texts, blind=False):
	"""Pangram copy task 3"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_pangrams_task(keymap, texts, 4, 6)

def task_13(keymap, hand, delay, blind=False):
	"""Uniform random task 7 with disappearing characters (3)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 3, blind=blind)

def task_14(keymap, texts, blind=False):
	"""Text copy task 4"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, (len(texts["minimal"])//6)*3, (len(texts["minimal"])//6)*4)

def task_15(keymap, hand, delay, blind=False):
	"""Uniform random task 8 (5)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 5, blind=blind)

def task_16(keymap, texts, blind=False):
	"""Pangram copy task 4"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_pangrams_task(keymap, texts, 6, 8)

def task_17(keymap, hand, delay, blind=False):
	"""Uniform random task 9 with disappearing characters (2)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 2, blind=blind)

def task_18(keymap, texts, blind=False):
	"""Text copy task 5"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, (len(texts["minimal"])//6)*4, (len(texts["minimal"])//6)*5)

def task_19(keymap, hand, delay, blind=False):
	"""Uniform random task 10 (3)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 3, blind=blind)

def task_20(keymap, texts, blind=False):
	"""Pangram copy task 5"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_pangrams_task(keymap, texts, 8, 10)

def task_21(keymap, hand, delay, blind=False):
	"""Uniform random task 11 with disappearing characters (2)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, True, 2, blind=blind)

def task_22(keymap, texts, blind=False):
	"""Text copy task 6"""
	# return if blind as there are no random elements in this task
	if blind:
		return [], "blind", False
	return copy_text_task(keymap, texts, (len(texts["minimal"])//6)*5, len(texts["minimal"]))

def task_23(keymap, hand, delay, blind=False):
	"""Uniform random task 12 (3)"""
	task = prepare_uniform_random_task(keymap, hand, 2)
	return uniform_random_task(keymap, task, delay, False, 3, blind=blind)

def task_24(keymap, diff, blind=False):
	"""Game task"""
	# return if blind as the random elements in the game are beyond control
	if blind:
		return [], "blind", False
	# introduce the game task
	if curses.wrapper(utils.wait_screen, keymap.start_game_prompt, True)[0]:
		return [], "game", False
	# start the game
	import alien_shower
	num_ships = 10
	sky_height = 11
	num_missiles = 2
	# imitate help in german
	if keymap == german_keymap:
		# show correct values in help
		if diff == "easy":
			num_ships = 5
			num_missiles = 2
		if diff == "normal":
			num_ships = 5
			num_missiles = 2
		if diff == "hard":
			num_ships = 10
			num_missiles = 2
		if diff == "brainfuck":
			num_ships = 10
			num_missiles = 3
		curses.wrapper(show_german_game_help, num_missiles, num_ships)
		try:
			alien_shower.run(diff, num_ships, sky_height, num_missiles, no_help=True)
		except argparse.ArgumentTypeError as error:
			print("The game task failed with the following error message:")
			print(error)
	else:
		try:
			alien_shower.run(diff, num_ships, sky_height, num_missiles)
		except argparse.ArgumentTypeError as error:
			print("The game task failed with the following error message:")
			print(error)
	return [], "game", False

def task_wrapper(_, myo_names, shared, prefix, task, hand, keymap):
	"""A wrapper to call tasks and stop recording after each."""
	if hand == "both":
		clap_func = partial(record.wait_for_sync_acc,
			prefix=prefix,
			max_lag=record.SYNC_MAX_LAG,
			min_acc_mag=record.SYNC_MIN_ACC_MAG,
			myo_names=myo_names,
			shared=shared
		)
		if curses.wrapper(utils.wait_screen, keymap.sync_prompt, True, run_while_waiting=clap_func)[0]:
			return [], None, False
	os.system("clear")
	result, task_type, stop_all = task()
	shared.stop_flag.set()
	return result, task_type, stop_all


### initialize and run functions ###

def get_standard_task_mapping():
	"""Return the task dictionaries gathered by task type."""
	return {
		"uniform": {1: task_1, 3: task_3, 5: task_5, 7: task_7, 9: task_9, 11: task_11,
			13: task_13, 15: task_15, 17: task_17, 19: task_19, 21: task_21, 23: task_23},
		"text": {2: task_2, 6: task_6, 10: task_10, 14: task_14, 18: task_18, 22: task_22},
		"pangram": {4: task_4, 8: task_8, 12: task_12, 16: task_16, 20: task_20},
		"game": {24: task_24}
	}

def init_task_dict(task_types, layout, hand, delay, game_difficulty):
	"""Initialize the tasks with their respective arguments and return them as dictionary."""
	# load keymap
	keymap = english_keymap if layout == "english" else german_keymap
	# load texts
	texts = {}
	if "text" in task_types or "pangram" in task_types:
		texts = utils.load_texts(layout)

	# initialize tasks
	mapping = get_standard_task_mapping()
	args_mapping = {
		"uniform": (keymap, hand, delay),
		"text": (keymap, texts),
		"pangram": (keymap, texts),
		"game": (keymap, game_difficulty),
	}
	for task_type in mapping:
		for task in mapping[task_type]:
			mapping[task_type][task] = partial(mapping[task_type][task], *args_mapping[task_type])

	# create a dictionary containing all initialized tasks
	considered = [mapping[task_type] for task_type in task_types]
	return {key: value for dic in considered for key, value in dic.items()}, keymap
	#return {key: value for k, dic in mapping.items() if k in task_types for key: value in dic.items()}

def run(seed, tasks, task_types, continue_at, task_args, record_args, user):
	"""Run the given tasks (either given as tasks or task types) and save the mistakes made."""
	random.seed(seed)

	# initialize all tasks and get the dictionary containing all tasks required by the argument
	# task_types
	task_dict, keymap = init_task_dict(
		task_types,
		task_args["layout"],
		task_args["hand"],
		task_args["delay"],
		task_args["difficulty"]
	)
	# if the task types are given, the list of tasks needs to be changed (reduced) to only include
	# the tasks of the given types, if no task types are given this does not change the tasks list
	tasks = [t for t in tasks if t in task_dict] # keep task order

	# show intro
	if curses.wrapper(utils.wait_screen, keymap.intro_prompt, True)[0]:
		return

	try:
		# run tasks
		fileprefix = record.enumerate_prefix(record_args["prefix"] if record_args else "off-record")
		count = int(fileprefix.split("-")[-1])
		mistakes = {}
		current_task = ""
		# continue with the task stopped at earlier and do everything else exactly as normal
		index = 0
		blind = False
		while index < len(tasks):
			if continue_at and continue_at != tasks[index]:
				blind = True
			else:
				blind = False
				continue_at = None
			current_random_state = random.getstate()
			current_task = f"{keymap.other_task}{tasks[index]}.\n"
			if index < len(tasks) - 1:
				current_task = current_task + f"{keymap.other_task_cont}{tasks[index+1]}.\n"

			# call record with the task wrapper
			if record_args and not blind:
				prefix = f"{fileprefix}.t{tasks[index]}"
				mistakes[f"Task {tasks[index]}"], task_type, stop_all = record.run(
					prefix=prefix,
					escape_key="Escape",
					record_keys=True,
					myo_devices=record_args["devices"],
					subscribe_args={
						"emg_mode": 3,
						"imu_mode": 1,
						"clf_state": 2,
						"battery": False
					},
					meta=types.SimpleNamespace(
						kb_model=record_args["kb_model"],
						kb_layout=record_args["kb_layout"],
						tag=record_args["tag"],
						pl_freq=record_args["pl_freq"]
					),
					main_task=partial(task_wrapper, prefix=prefix, task=task_dict[tasks[index]], hand=task_args["hand"], keymap=keymap)
				)
				# write additional meta data
				record.add_meta_data(prefix, {
					"id": {
						"user": user if user != -1 else count,
						"collection": count,
					},
					"common": {
						"seed": seed,
						"typing_style": record_args["typing_style"],
						"task_type": task_type,
						"task_id": index + 1
					},
					"mistakes": mistakes[f"Task {tasks[index]}"],
					"notes": []
				})
			# call the tasks directly without recording
			else:
				mistakes[f"Task {tasks[index]}"], _, stop_all = task_dict[tasks[index]](blind=blind)

			# continue if blind as there are no more random elements
			if blind:
				index += 1
				continue
			next_prompt = keymap.end_task_prompt
			wrapper_func = utils.wait_screen
			half = len(tasks)//2 - 1
			# break if the last task is done
			if index == len(tasks) - 1:
				current_task = keymap.last_task
				next_prompt = keymap.outro_prompt + current_task + "\n" + keymap.repeat_prompt + "\n"
			# ask the user to change his posture
			elif (index + 1) % 4 == 0:
				next_prompt = keymap.end_task_with_posture_change_prompt
				wrapper_func = partial(utils.wait_screen, bold_header=True)
			# encourage the user to take an extended break at half time
			elif len(tasks) > 8 and (index == half and (index + 1) % 4 != 0) or (index == half + 1 and index % 4 == 0):
				next_prompt = keymap.end_task_with_extended_break_prompt

			# break if a keyboard exception occurs
			if stop_all:
				break

			# wait for user input to start the next task
			stop, go_back, _ = curses.wrapper(wrapper_func, next_prompt, footer_len=2)
			if stop:
				break
			if go_back:
				random.setstate(current_random_state)
				continue
			index += 1

	except (KeyboardInterrupt, SystemExit):
		# stop with a last thank you message
		curses.wrapper(utils.wait_screen, keymap.outro_prompt + current_task)

	# reprint it to retrieve the task if the recording was stopped early
	print(keymap.outro_prompt + current_task)
	if not record_args:
		print(mistakes)


def invalid_task_check(task_dict):
	class InvalidTaskCheck(argparse.Action):
		"""Check for invalid tasks."""
		def __call__(self, parser, args, values, option_string=None):
			valid_tasks = " ".join(map(str, sorted(task_dict.keys())))
			for i in values:
				if not i in task_dict:
					raise argparse.ArgumentTypeError(f"Found invalid task {i}. Valid tasks are either of: {valid_tasks}")
			setattr(args, self.dest, values)
	return InvalidTaskCheck

def main():
	"""Parse the arguments."""
	# set log level
	logging.basicConfig(level=logging.INFO, format="[%(processName)s] %(message)s")

	# get the tasks available
	mapping = get_standard_task_mapping()
	task_dict = {key: value for dic in mapping.values() for key, value in dic.items()}

	# standard arguments
	parser = argparse.ArgumentParser(description="Run the data recording tasks.")
	task_group = parser.add_mutually_exclusive_group()
	task_group.add_argument("-t", "--tasks", dest="tasks", type=int, nargs="+", default=list(range(1, len(task_dict)+1)), action=invalid_task_check(task_dict), help="the tasks to run (WARNING: do not use this to continue a data collection - use continue_at instead - as it will change the seed for the uniform tasks!)")
	task_group.add_argument("-p", "--task_types", dest="task_types", nargs="+", choices=["text", "pangram", "uniform", "game"], default=["text", "pangram", "uniform", "game"], help="the types of tasks to run (WARNING: do not use this to continue a data collection - use continue_at instead - as it will change the seed for the uniform tasks!)")
	parser.add_argument("-c", "--continue_at", dest="continue_at", choices=list(range(1, len(task_dict)+1)), type=int, help="the task to continue the recording with (can be used to continue the data collection normally if it was stopped in between)")
	parser.add_argument("-a", "--hand", dest="hand", choices=["both", "left", "right", "none"], default="both", help="the hand(s) to record data from")
	parser.add_argument("-l", "--layout", dest="layout", choices=["german", "english"], default="german", metavar="", help="the keyboard layout (German QWERTZ or English US QWERTY)")
	parser.add_argument("-s", "--seed", dest="seed", type=int, default=0, metavar="", help="the random seed")
	parser.add_argument("-u", "--user", type=int, default=-1, metavar="", help="the custom user identifier")
	parser.add_argument("-d", "--delay", dest="delay", type=float, default=0.2, metavar="", help="the delay or minimal time for which a character is displayed (also a time between tasks)")
	# game argument
	game_group = parser.add_argument_group("game")
	game_group.add_argument("-g", "--difficulty", dest="difficulty", choices=["easy", "normal", "hard", "brainfuck", "custom"], default="custom", help="the game task difficulty which predefines the number of ships, sky height, missiles and speed")

	# record arguments
	subparsers = parser.add_subparsers(dest="rec")
	parser_rec = subparsers.add_parser("record", help="record data")
	parser_rec.add_argument("typing_style", choices=["touch_typing", "hybrid"], help="the user's typing style")
	parser_rec.add_argument("--path", default="../train-data/", help="the data path, default: \"%(default)s\"")
	parser_rec.add_argument("--left_tty", metavar="TTY", required="right" not in sys.argv and "none" not in sys.argv, help="the Myo dongle TTY for the Myo worn on the left arm")
	parser_rec.add_argument("--left_mac", metavar="MAC", required="right" not in sys.argv and "none" not in sys.argv, help="the Myo MAC address for the Myo worn on the left arm")
	parser_rec.add_argument("--right_tty", metavar="TTY", required="left" not in sys.argv and "none" not in sys.argv, help="the Myo dongle TTY for the Myo worn on the right arm")
	parser_rec.add_argument("--right_mac", metavar="MAC", required="left" not in sys.argv and "none" not in sys.argv, help="the Myo MAC address for the Myo worn on the right arm")
	# meta data arguments
	parser_rec.add_argument("--tag", default=None, help="the description of the recorded data, default (always included): \"Data Collection <NUMBER>\" where NUMBER is the amount of data recordings present in the given path")
	parser_rec.add_argument("--kb_model", choices=record.KB_CONFIG["models"].keys(), action=record.extract_action(record.KB_CONFIG["models"]), required=True, help="description of the physical keyboard")
	parser_rec.add_argument("--kb_layout", choices=record.KB_CONFIG["layouts"], default="us" if "english" in sys.argv else "de", help="the current keyboard layout, default: same as <layout>")
	parser_rec.add_argument("--pl_freq", type=float, default=50.0, help="the power line frequency in Hz")

	args = parser.parse_args()

	task_args = {
		"layout": args.layout,
		"hand": args.hand,
		"delay": args.delay,
		"difficulty": args.difficulty
	}
	if args.rec:
		if args.hand == "right":
			devs = [(args.right_tty, args.right_mac, "right")]
		elif args.hand == "left":
			devs = [(args.left_tty, args.left_mac, "left")]
		elif args.hand == "both":
			devs = [(args.left_tty, args.left_mac, "left"), (args.right_tty, args.right_mac, "right")]
		elif args.hand == "none":
			devs = []
		# default tag for data collection counts the recorded hybrid, touch typing and hunt and peck
		# data samples to increment the total data collection sample index
		path = args.path if args.path.endswith("/") else args.path + "/"
		record_args = {
			"prefix": path + "record",
			"devices": devs,
			"tag": args.tag if args.tag else "Data Study: Inferring Keystrokes from Myo Armband Sensors",
			"kb_model": args.kb_model,
			"kb_layout": args.kb_layout,
			"typing_style": args.typing_style,
			"pl_freq": args.pl_freq
		}
	else:
		record_args = None

	run(args.seed, args.tasks, args.task_types, args.continue_at, task_args, record_args, args.user)

if __name__ == "__main__":
	main()
