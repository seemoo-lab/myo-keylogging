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
import curses
import os
import types
from functools import partial
import sys
import logging
import subprocess

import german_keymap
import english_keymap
import utils

import record

total_task_num = 0
current_task_num = 0
coins = 0

### password task specific functions ###

def turn_coins(coins):
	coin_symbols = [chr(0x1F312), chr(0x1F311), chr(0x1F318)]
	print_coins = ""
	for i in range(coins):
		print_coins += coin_symbols[i % 3]
	return print_coins

def pw_customization(_, stdscr, result, task, keymap, prompt):
	"""
	If return or space are typed, replace them with their respective symbols such that they resemble
	the task characters to copy (so as not to confuse the user).
	Let the task disappear when the first key is hit if the flag is set to do so and stop the
	subtask automatically as soon as all characters have been typed.
	"""

	# shortcut as everything else relies on result
	if not result:
		return False

	# end on return
	if result[-1] == "\n":
		result.pop()
		if result == task:
			success = 3
		else:
			success = 2
		# give the user a glimpse of the characters typed
		# printed red if the password was wrong, green if it was right
		# should give feedback without disturbing the flow (otherwise adjust the value)
		stdscr.clear()
		utils.print_pw_task(stdscr, prompt, "".join(task), "".join(result), success, additional_prompt=keymap.additional_pw_prompt, reprint_task=True)
		stdscr.refresh()
		while True:
			key = stdscr.getch()
			if key == 10:
				return True # break from task loop on enter

	# change the appearance of space to fit the task
	if result[-1] == " " and not " " in task:
		result[-1] = chr(9251)
		stdscr.clear()
		utils.print_pw_task(stdscr, prompt, "".join(task), "".join(result), additional_prompt=keymap.additional_pw_prompt)

	# print wildcards (echo is deactivated)
	stdscr.clear()
	utils.print_pw_task(stdscr, prompt, "".join(task), "*" * len(result), additional_prompt=keymap.additional_pw_prompt)

	return False

def pw_task(keymap, tasks, num_repetitions, task_tag, blind=False):
	passwords = tasks.copy()
	stop_all = False
	try:
		# start screen
		global current_task_num
		current_task_num += 1
		task_count = f"\n\n\n- {current_task_num}/{total_task_num} -"
		mistakes = {}
		cuts = []
		if not blind:
			task_type = ""
			if task_tag.startswith("random"):
				task_type = keymap.random_task
			elif task_tag.startswith("pwgen"):
				task_type = keymap.pwgen_task
			elif task_tag.startswith("xkcd"):
				task_type = keymap.xkcd_task
			elif task_tag.startswith("insecure"):
				task_type = keymap.insecure_task
			discontinue, _, num_keys = curses.wrapper(utils.wait_screen,
				task_type + keymap.start_pw_task_prompt + task_count,
				True
			)
			if discontinue:
				return mistakes, task_tag, passwords, stop_all, cuts
		cuts.append(time.time() - record.START_TIME)

		# run task
		stop = False
		custom_print_func = partial(utils.print_pw_task,
			additional_prompt=keymap.additional_pw_prompt)
		run_subtask_partial = partial(utils.run_subtask,
			keymap=keymap, prompt=keymap.copy_pw_prompt, custom_print_func=custom_print_func)
		custom_check_partial = partial(pw_customization)
		rep_counter = 0
		if not blind:
			while tasks and not stop:
				task = tasks[-1]
				run_subtask = partial(run_subtask_partial,
					custom_check_func=partial(custom_check_partial)
				)
				stop, result, add_to_num_keys = curses.wrapper(run_subtask, task=list(task))
				cuts.append(time.time() - record.START_TIME)
				prior_mistakes = len(mistakes)
				utils.gather_mistakes(mistakes, list(task), result, num_keys)
				num_keys += add_to_num_keys
				if len(mistakes) <= prior_mistakes and len(result) == len(task):
					rep_counter += 1
				# repeat task until it has been typed correctly num_repetition times
				if rep_counter == num_repetitions:
					rep_counter = 0
					tasks.pop()
					# randomly distribute coins for hard tasks
					coin_toss = random.random()
					if not task_tag.startswith("insecure"):
						global coins
						if (coin_toss >= 0.9 and current_task_num > 2) or (current_task_num == total_task_num//2 and coins < 10):
							reward = 3
							message = keymap.task_changes_extra_coin
						elif coin_toss >= 0.2:
							reward = 1
							message = keymap.task_changes_coin
						else:
							reward = 0
							message = keymap.task_changes
						coins += reward
						print_coins = turn_coins(coins)
						if reward > 0:
							print_coins = print_coins[:-reward] + "\n+ " + print_coins[-reward:]
						curses.wrapper(utils.wait_screen, print_coins + "\n\n" + message, True)
		else: # simulate random call for blind run
			for _ in tasks:
				coin_toss = random.random()

	except (KeyboardInterrupt, SystemExit):
		stop_all = True
	return mistakes, task_tag, passwords, stop_all, cuts

def prepare_random_pw_task(keymap, common, hand, num_pws, word_length):
	"""
	Prepare the task with the given number of passwords of the given length. Include 10% random
	shifted character and choose random characters from the keymap.
	The keymap determines the characters used for the task and the presumed keyboard layout.
	If only one hand is given only the respective characters typed with this hand are included
	(except for space and enter, which are always included).
	"""
	if common:
		left_keys = keymap.common_left_keys
		right_keys = keymap.common_right_keys
	else:
		left_keys = keymap.left_keys
		right_keys = keymap.right_keys

	# choose 10% random keys per hand and use their shift-modified version
	# all others are non-shifted
	left_shift = set()
	right_shift = set()
	for _ in range((len(left_keys) + len(right_keys))//10):
		left_shift.add(random.randint(0, len(left_keys)-1))
		right_shift.add(random.randint(0, len(right_keys)-1))

	# get the characters to type from the list of keys (considering shift)
	chars_left = [k[1] if i in left_shift else k[0] for i, k in enumerate(left_keys)]
	chars_right = [k[1] if i in right_shift else k[0] for i, k in enumerate(right_keys)]

	if hand == "left":
		copy = chars_left
	elif hand == "right":
		copy = chars_right
	else:
		copy = chars_left + chars_right
	# add space
	copy.append(chr(9251))

	# choose as many random characters as necessary
	task = []
	for _ in range(num_pws * word_length):
		task.append(random.choice(copy))

	# split chars array into pws array with pws of length word_length
	task = ["".join(a) for a in [task[i:i+word_length] for i in range(0, len(task), word_length)]]
	return task

def prepare_pwgen_pw_task(keymap, common, hand, num_pws, word_length, seed, iteration):
	"""
	Prepare the task with the given number of passwords of the given length. Use pwgen to generate
	passwords, replacing invalid chars in respect to the keymap with valid chars.
	The keymap determines the characters used for the task and the presumed keyboard layout.
	If only one hand is given only the respective characters typed with this hand are included
	(except for space and enter, which are always included).
	"""
	if common:
		left_keys = keymap.common_left_keys
		right_keys = keymap.common_right_keys
	else:
		left_keys = keymap.left_keys
		right_keys = keymap.right_keys

	# load keymap
	all_chars_left = [char for tuples in left_keys for char in tuples]
	all_chars_right = [char for tuples in right_keys for char in tuples]
	if hand == "left":
		all_chars = all_chars_left
	elif hand == "right":
		all_chars = all_chars_right
	else:
		all_chars = all_chars_left + all_chars_right
	all_chars = all_chars + [chr(9251)]

	# call pwgen to generate passwords
	# -c: enforce at least one capital
	# -n: enforce at least one number
	# -y: enforce at least one symbol
	# though due to above replacement the end result may look different
	pwgen_cmd = ["pwgen", "-cny", f"{word_length}", f"{num_pws}", "-H", f"seed#{seed + iteration}"]
	pws = subprocess.run(pwgen_cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
	pws = pws.replace(" ", chr(9251))

	# get all chars not present in the keymap but in the passwords
	bad_chars = []
	for char in pws:
		if char not in all_chars:
			bad_chars.append(char)

	# get all chars present in the keymap but not in the passwords
	missing_chars = []
	for char in all_chars:
		if char not in pws:
			missing_chars.append(char)
	if not missing_chars:
		for _ in range(bad_chars):
			missing_chars.append(random.choice(all_chars))

	# replace all chars not present in the keymap with random chars not present in the passwords
	for bad_char in bad_chars:
		pws = pws.replace(bad_char, random.choice(missing_chars))
		#print(f"{bad_char} randomly replaced")

	# remove spaces between pws and return a list of pws
	task = [pws[i:i+word_length] for i in range(0, len(pws), word_length+1)]
	#task = [char for word in task for char in word]
	return task

def prepare_insecure_pw_task(keymap, hand, dictionary, num_pws):
	"""
	Prepare the task with the given number of passwords. Extract them from a list of common
	passwords as provided by SecLists (https://github.com/danielmiessler/SecLists).
	If only one hand is given no password is returned.
	"""
	if hand != "both":
		return []

	task = set()
	while len(task) < num_pws:
		random.shuffle(dictionary)
		task.add(dictionary.pop())
	return list(task)

def prepare_xkcd_pw_task(keymap, hand, dictionary, num_words, num_pws):
	"""
	Prepare the task with the given number of passwords generated by xkcdpass. The passwords will
	be generated from scratch every time, which makes this task unrepeatable.
	If only one hand is given no password is returned.
	"""
	if hand != "both":
		return []

	task = set()
	while len(task) < num_pws:
		pw = " ".join([random.choice(dictionary) for _ in range(num_words)])
		task.add(pw)

	# not reproducible, thus not usable:
#	pwgen_cmd = ["xkcdpass", f"-c{num_pws}", "-d", "\"\"", "-C", "first"]
#	task = subprocess.run(pwgen_cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
	return list(task)

def get_dictionary(name):
	with open(f"english/{name}", "r") as file_obj:
		pws = file_obj.read().splitlines()
	pws.pop(0) # remove comment
	return pws


### task definition functions ###

def task_random_pw(keymap, common, hand, seed, iteration, num_chars=8, num_repetitions=6, num_passwords=4, blind=False):
	task = prepare_random_pw_task(keymap, common, hand, num_passwords, num_chars)
	return pw_task(keymap, task, max(num_repetitions, 1), "random", blind=blind)

def task_pwgen_pw(keymap, common, hand, seed, iteration, num_chars=8, num_repetitions=6, num_passwords=4, blind=False):
	task = prepare_pwgen_pw_task(keymap, common, hand, num_passwords, num_chars, seed, iteration)
	return pw_task(keymap, task, max(num_repetitions, 1), "pwgen", blind=blind)

def task_xkcd_pw(keymap, common, hand, seed, iteration, dictionary, num_words=6, num_repetitions=6, num_passwords=4, blind=False):
	task = prepare_xkcd_pw_task(keymap, hand, dictionary, num_words, num_passwords)
	return pw_task(keymap, task, max(num_repetitions, 1), f"xkcd", blind=blind)

def task_insecure_pw(keymap, common, hand, seed, iteration, dictionary, num_repetitions=6, num_passwords=4, blind=False):
	task = prepare_insecure_pw_task(keymap, hand, dictionary, num_passwords)
	return pw_task(keymap, task, max(num_repetitions, 1), f"insecure", blind=blind)

def task_wrapper(_, myo_names, shared, prefix, task, hand, keymap, seed, iteration):
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
			return [], None, False, []
	os.system("clear")
	result, task_type, passwords, stop_all, cuts = task(seed=seed, iteration=iteration)
	shared.stop_flag.set()
	return result, task_type, passwords, stop_all, cuts



def run(seed, continue_at, task_args, record_args, user, count):
	"""Run the given tasks (either given as tasks or task types) and save the mistakes made."""
	# if there is no recording and count is not given, initialize count randomly
	random.seed(seed)

	# load dictionaries
	insecure_dict = get_dictionary("seclists_passwords")
	eff_dict = get_dictionary("eff_long")

	# define tasks
	keymap = english_keymap if task_args["layout"] == "english" else german_keymap

	pwn = max(1, task_args["num_passwords"])
	hand = task_args["hand"]
	num_reps = task_args["num_repetitions"]
	tasks = []
	for i in range(0,3):
		if i < 2:
			common = True
		else:
			common = False
		tasks.extend([
			partial(task_random_pw,
				keymap, common, hand, num_chars=8, num_repetitions=num_reps, num_passwords=pwn),
			partial(task_pwgen_pw,
				keymap, common, hand, num_chars=8, num_repetitions=num_reps, num_passwords=pwn),
			partial(task_xkcd_pw,
				keymap, common, hand, dictionary=eff_dict, num_repetitions=num_reps, num_passwords=pwn),
			partial(task_insecure_pw,
				keymap, common, hand, dictionary=insecure_dict, num_repetitions=num_reps, num_passwords=pwn),
		])
	global total_task_num
	total_task_num = len(tasks)
	# get the index to change the seed (after recording with this index)
	change_index = len(tasks) - len(tasks)/3 - 1

	# show intro
	if curses.wrapper(utils.wait_screen, keymap.intro_prompt, True)[0]:
		return

	try:
		# run tasks
		fileprefix = record.enumerate_prefix(record_args["prefix"] if record_args else "off-record")
		count = count if count != -1 else int(fileprefix.split("-")[-1])
		print(f"Collection {count}")
		mistakes = {}
		cuts = {}
		current_task = ""
		# continue with the task stopped at earlier and do everything else exactly as normal
		index = 0
		blind = False
		done_prompt = ""
		while index < len(tasks):
			if continue_at and continue_at != index:
				blind = True
			else:
				blind = False
				continue_at = None
			current_random_state = random.getstate()
			current_task = f"{keymap.other_task}{index}.\n"
			if index < len(tasks) - 1:
				current_task = current_task + f"{keymap.other_task_cont}{index+1}.\n"

			# call record with the task wrapper
			if record_args and not blind:
				prefix = f"{fileprefix}.t{index}"
				mistakes[f"Password Task {index}"], task_type, passwords, stop_all, cuts[f"Password Task {index}"] = record.run(
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
					main_task=partial(task_wrapper, prefix=prefix, task=tasks[index], hand=hand, keymap=keymap, seed=seed, iteration=index)
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
						"task_id": index
					},
					"task": {
						"repetitions": num_reps,
						"passwords_per_task": pwn,
						"passwords": passwords
					},
					"mistakes": mistakes[f"Password Task {index}"],
					"simple_cuts": cuts[f"Password Task {index}"],
					"notes": []
				})
			# call the tasks directly without recording
			else:
				mistakes[f"Password Task {index}"], _, _, stop_all, cuts[f"Password Task {index}"] = tasks[index](seed=seed, iteration=index, blind=blind)

			# change seed
			if index == change_index:
				seed += count + 1
				random.seed(seed)

			# continue if blind as there are no more random elements
			if blind:
				index += 1
				continue
			next_prompt = keymap.end_task_prompt
			wrapper_func = utils.wait_screen
			# break if the last task is done
			if index == len(tasks) - 1:
				current_task = keymap.last_task
				jellybears = "\n" + keymap.treats_prompt
				done_prompt = keymap.outro_prompt + current_task + "\n" + keymap.orb_prompt + f"{coins} " + turn_coins(coins) + jellybears + "\n"
				next_prompt = done_prompt + "\n" + keymap.repeat_prompt + "\n"
			# ask the user to change his posture
			#elif (index + 1) % 4 == 0:
			#	next_prompt = keymap.end_task_with_posture_change_prompt
			#	wrapper_func = partial(utils.wait_screen, bold_header=True)

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
		print(done_prompt)

	except (KeyboardInterrupt, SystemExit):
		pass

	if done_prompt == "":
		# stop with a last thank you message
		jellybears = "\n" + keymap.treats_prompt if coins > 10 else ""
		bye_string = keymap.outro_prompt + keymap.orb_prompt + f"{coins} " + turn_coins(coins) + jellybears + "\n\n" + current_task
		curses.wrapper(utils.wait_screen, bye_string)
		# reprint it to retrieve the task if the recording was stopped early
		print(bye_string)

	if not record_args:
		print(mistakes)
		print(cuts)



def main():
	"""Parse the arguments."""
	# set log level
	logging.basicConfig(level=logging.INFO, format="[%(processName)s] %(message)s")

	# standard arguments
	parser = argparse.ArgumentParser(description="Run the password recording tasks.")
	parser.add_argument("-r", "--repetitions", dest="num_repetitions", type=int, default=4, metavar="", help="the number of correct repetitions for typing the passwords required for continuing")
	parser.add_argument("-n", "--npw", dest="num_passwords", type=int, default=2, metavar="", help="the number of different passwords to type in the individual tasks (common tasks will be double)")
	parser.add_argument("-c", "--continue_at", dest="continue_at", type=int, help="the task to continue the recording with (can be used to continue the data collection normally if it was stopped in between)")
	parser.add_argument("-a", "--hand", dest="hand", choices=["both", "left", "right", "none"], default="both", help="the hand(s) to record data from")
	parser.add_argument("-l", "--layout", dest="layout", choices=["german", "english"], default="german", metavar="", help="the keyboard layout (German QWERTZ or English US QWERTY)")
	parser.add_argument("-s", "--seed", dest="seed", type=int, default=20, metavar="", help="the random seed")
	parser.add_argument("-i", "--collection", type=int, default=-1, metavar="", help="the custom collection identifier")
	parser.add_argument("-u", "--user", type=int, default=-1, metavar="", help="the custom user identifier")
	#parser.add_argument("-d", "--delay", dest="delay", type=float, default=0.2, metavar="", help="the delay or minimal time for which a character is displayed (also a time between tasks)")

	# record arguments
	subparsers = parser.add_subparsers(dest="rec")
	parser_rec = subparsers.add_parser("record", help="record data")
	parser_rec.add_argument("typing_style", choices=["touch_typing", "hybrid"], help="the user's typing style")
	parser_rec.add_argument("--path", default="../test-data/", help="the data path, default: \"%(default)s\"")
	parser_rec.add_argument("--left_tty", metavar="TTY", required="right" not in sys.argv and "none" not in sys.argv, help="the Myo dongle TTY for the Myo worn on the left arm")
	parser_rec.add_argument("--left_mac", metavar="MAC", required="right" not in sys.argv and "none" not in sys.argv, help="the Myo MAC address for the Myo worn on the left arm")
	parser_rec.add_argument("--right_tty", metavar="TTY", required="left" not in sys.argv and "none" not in sys.argv, help="the Myo dongle TTY for the Myo worn on the right arm")
	parser_rec.add_argument("--right_mac", metavar="MAC", required="left" not in sys.argv and "none" not in sys.argv, help="the Myo MAC address for the Myo worn on the right arm")
	# meta data arguments
	parser_rec.add_argument("--tag", default=None, help="the description of the recorded data, default: \"Data Study: Inferring Keystrokes from Myo Armband Sensors\"")
	parser_rec.add_argument("--kb_model", choices=record.KB_CONFIG["models"].keys(), action=record.extract_action(record.KB_CONFIG["models"]), required=True, help="description of the physical keyboard")
	parser_rec.add_argument("--kb_layout", choices=record.KB_CONFIG["layouts"], default="us" if "english" in sys.argv else "de", help="the current keyboard layout, default: same as <layout>")
	parser_rec.add_argument("--pl_freq", type=float, default=50.0, help="the power line frequency in Hz")

	args = parser.parse_args()

	task_args = {
		"layout": args.layout,
		"hand": args.hand,
		"num_repetitions": args.num_repetitions,
		"num_passwords": args.num_passwords,
		#"delay": args.delay,
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

	run(args.seed, args.continue_at, task_args, record_args, args.user, args.collection)

if __name__ == "__main__":
	main()
