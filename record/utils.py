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

import random
import curses
import textwrap
from itertools import zip_longest

def print_task(stdscr, prompt, task, result=""):
	"""Print a task in curses with a free line between the given prompt, the task and the result."""
	_, scr_width = stdscr.getmaxyx()
	# -3 and " " + rest to leave a space left and right of the screen for better readability
	task_array = [textwrap.dedent(line) for line in textwrap.wrap("".join(task), scr_width-3, drop_whitespace=False)]
	result_array = [textwrap.dedent(line) for line in textwrap.wrap("".join(result), scr_width-3, drop_whitespace=False)]
	stdscr.addstr(1, 0, " " + prompt + "\n\n " + "\n ".join(task_array) + "\n\n " + "\n ".join(result_array))

def print_pw_task(stdscr, prompt, task, result="", success=0, additional_prompt="", reprint_task=False):
	"""Print a task in curses with a free line between the given prompt, the task and the result."""
	curses.use_default_colors() # use colors as used per default in the terminal
	curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
#	scr_height, scr_width = stdscr.getmaxyx()
#	# -3 and " " + rest to leave a space left and right of the screen for better readability
#	task_array = [textwrap.dedent(line) for line in textwrap.wrap("".join(task), scr_width-3, drop_whitespace=False)]
#	result_array = [textwrap.dedent(line) for line in textwrap.wrap("".join(result), scr_width-3, drop_whitespace=False)]
	stdscr.addstr(1, 0, " " + "".join(prompt))
	stdscr.addstr(3, 0, " " + "".join(task), curses.A_BOLD)
	stdscr.addstr(8, 0, " " + additional_prompt)
	stdscr.addstr("\n " + "password:")
	stdscr.addstr("\n " + "".join(result), curses.color_pair(success))
	if reprint_task:
		stdscr.addstr("\n " + "".join(task))
	#stdscr.addstr(scr_height-3, 0, " " + "password:")
	#stdscr.addstr(scr_height-2, 0, " " + "".join(result_array), curses.color_pair(success))


def increment_keys(num_keys, key):
	if key not in (-1, 27):
		if curses.keyname(key).startswith(b'M-'):
			num_keys += 0.5
		else:
			num_keys += 1
	return num_keys

def wait_screen(stdscr, prompt, bold_header=False, footer_len=1, run_while_waiting=None):
	"""Use curses to wait for user input to start the current task (or end it)."""
	curses.use_default_colors() # use colors as used per default in the terminal
	curses.curs_set(0) # hide cursor
	curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
	curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
	curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
	_, scr_width = stdscr.getmaxyx()
	stdscr.clear()
	num_keys = 0
	while True:
		prompt_array = prompt.split("\n")
		# use textwrap for pretty-printing the header
		header = "\n ".join(textwrap.wrap(prompt_array[0], scr_width-3))
		if bold_header:
			if header in ("Verschwindende Zufallszeichen", "Disappearing Random Characters"):
				stdscr.addstr(1, 0, " " + header + "\n", curses.A_BOLD | curses.color_pair(2))
			elif header in ("Bleibende Zufallszeichen", "Lasting Random Characters"):
				stdscr.addstr(1, 0, " " + header + "\n", curses.A_BOLD | curses.color_pair(5))
			elif header == "Text":
				stdscr.addstr(1, 0, " " + header + "\n", curses.A_BOLD | curses.color_pair(6))
			elif header in ("Pangramme", "Pangrams"):
				stdscr.addstr(1, 0, " " + header + "\n", curses.A_BOLD | curses.color_pair(3))
			else:
				stdscr.addstr(1, 0, " " + header + "\n", curses.A_BOLD)
		else:
			stdscr.addstr(1, 0, " " + header + "\n")
		# use textwrap for pretty-printing the text
		if footer_len:
			rest = prompt_array[1:-footer_len]
		else:
			rest = prompt_array[1:]
		for line in rest:
			stdscr.addstr(" " + "\n ".join(textwrap.wrap(line, scr_width-3)) + "\n")
		# run a given function once
		if run_while_waiting:
			stdscr.refresh()
			run_while_waiting()
			curses.flushinp()
			run_while_waiting = None
		# add the footer
		if footer_len:
			for line in prompt_array[-footer_len:]:
				stdscr.addstr(" " + "\n ".join(textwrap.wrap(line, scr_width-3)) + "\n", curses.A_ITALIC)
		key = stdscr.getch()
		if key == 27:
			return True, False, num_keys
		if key == 114: #r
			return False, True, num_keys
		num_keys = increment_keys(num_keys, key)
		if key == 10:
			break
	curses.curs_set(1) # show cursor
	return False, False, num_keys

def run_subtask(stdscr, keymap, prompt, task, custom_check_func=None, custom_print_func=None):
	"""
	Use curses to present the help and characters to type for a given subtask and handle the input.
	A custom function may provide additional functionality or a stopping condition.
	"""
	# don't wait for input (while showing a black input screen)
	stdscr.nodelay(True)
	# use colors as used per default in the terminal
	curses.use_default_colors()

	if not custom_print_func:
		custom_print_func = print_task

	# print help
	stdscr.clear()
	custom_print_func(stdscr, prompt, task)

	# run the subtask
	result = []
	bytepart = None
	stop = False
	num_keys = 0
	while True:
		key = stdscr.getch()
		if key != -1:
			# if escape is pressed, exit
			if key == 27:
				stop = True
				break
			num_keys = increment_keys(num_keys, key)
			# handle backspace
			if curses.keyname(key) in (b'^?', b'KEY_BACKSPACE'):
				if result:
					result.pop()
				stdscr.clear()
				custom_print_func(stdscr, prompt, task, "".join(result))
			# ignore backspace and pipe
			elif key in (92, 124):
				pass
			# don't handle certain keys
			elif curses.keyname(key) in (
				b'KEY_LEFT',
				b'KEY_UP',
				b'KEY_RIGHT',
				b'KEY_DOWN',
				b'KEY_HOME',
				b'KEY_END',
				b'KEY_DC',
				b'KEY_STAB'
			):
				pass
			else:
				# those keys that start with M-{*} when calling curses.keyname(key)
				# seem to be the ones producing double input
				# some of them (umlaut) can be fixed by combining the first byte of the
				# first recognized key with the last byte of the second recognized key
				# others are correct if the first two bytes are dumped
				encoded_key = str(chr(key)).encode("utf8")
				if curses.keyname(key).startswith(b'M-'):
					if bytepart:
						new_key = bytepart + encoded_key[-1:]
						new_key = new_key
						if not any(str(new_key.decode("utf8")) in tuples for tuples in keymap.keys):
							new_key = encoded_key
							if not any(str(new_key.decode("utf8")) in tuples for tuples in keymap.keys):
								bytepart = None
								continue
						encoded_key = new_key
						bytepart = None
					else:
						bytepart = encoded_key[:1]
						continue
				result.append(encoded_key.decode("utf8"))
			if custom_check_func(key, stdscr, result, task, keymap, prompt):
				break
	return stop, result, num_keys

def gather_mistakes(mistakes, task, result, num_keys):
	"""Check for mistakes in the given result and gather them in the given mistakes array."""
	for i, (expected, given) in enumerate(zip_longest(task, result, fillvalue="")):
		if expected != given and not given == "":
			if given == chr(8629):
				given = "\\n"
			if given == chr(9251):
				given = " "
			mistakes[num_keys + i] = {"expected": expected, "given": given}


def split_file_content(file_content, offset=False):
	"""
	Parse the given file content (split at empty line).
	Offset will ignore a header above each text sample.
	"""
	extracts = []
	start = 1 if offset else 0
	for end, text in enumerate(file_content):
		if text == "":
			extracts.append(file_content[start:end])
			start = end + 2 if offset else end + 1
	rest = file_content[start:]
	if rest:
		extracts.append(rest)
	return extracts

def load_texts(layout):
	"""Load the texts."""
	minimal_text = []
	with open(layout + "/minimal", "r") as file_obj:
		minimal_text = file_obj.read().splitlines()
	minimal_text = split_file_content(minimal_text, True)
	random.shuffle(minimal_text)

	fun_text = []
	with open(layout + "/fun_facts", "r") as file_obj:
		fun_text = file_obj.read().splitlines()
	fun_text = split_file_content(fun_text, True)
	random.shuffle(fun_text)

	pangrams = []
	with open(layout + "/pangrams", "r") as file_obj:
		pangrams = file_obj.read().splitlines()
	pangrams = split_file_content(pangrams)
	random.shuffle(pangrams)

	return {"minimal": minimal_text, "fun_facts": fun_text, "pangrams": pangrams}

def convert_to_char(key_symbol):
	# ignore characters not considered for learning
	if key_symbol in IGNORED_KEYS:
		return chr(0x25a0)
	# replace considered characters not given as symbol
	if key_symbol in CONSIDERED_KEYS:
		return CONSIDERED_KEYS[key_symbol]
	return key_symbol

IGNORED_KEYS = [
	"ISO_Level3_Shift", "Super_L", "Super_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
	"Down", "Up", "Left", "Right",
	"dead_acute", "dead_circumflex", "grave",
	"backslash",
	"Delete", "Home", "Tab"
]

CONSIDERED_KEYS = {
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
