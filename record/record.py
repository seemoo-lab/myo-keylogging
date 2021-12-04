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

"""Record data in parallel from multiple Myos and the keyboard."""
import argparse
import multiprocessing as mp
import types
import contextlib
import csv
import time
import json
import pathlib
import functools
import sys
import termios
import logging
import os
import signal
import threading
import queue
import collections

import xkeyboard
import myo_raw

LOG = logging.getLogger(__name__)

START_TIME = time.time()

TIMEOUT = 3

SYNC_MIN_ACC_MAG = 6144 # = 3 g
SYNC_MAX_LAG = 0.1 # seconds

KB_CONFIG = {
	"models": {
		"Cherry": "Cherry G80 DE ISO",
		"TADA68_DE": "TADA68 DE ISO",
		"TADA68_US": "TADA68 US ISO",
		"ClassicTP": "Classic ThinkPad UK ISO"
	},
	"layouts": ["de", "us"]
}

def _rep(name, num):
	"""Repeat the given string appended with an increasing number (starting with zero)."""
	return [name + str(i) for i in range(num)]

def _flatten_tuple(iterable):
	"""Flatten a tuple."""
	for item in iterable:
		if isinstance(item, tuple):
			yield from _flatten_tuple(item)
		else:
			yield item

def _write_data(timestamp, *args, csv_writer, offset=START_TIME):
	"""Write key and sensors data to a csv file storing relative instead of absolute timestamps"""
	csv_writer.writerow(_flatten_tuple((timestamp - offset,) + args))

def _record_acc_mag(timestamp, quat, acc, gyro, acc_queue, identifier, offset=START_TIME):
	"""Put the timestamp and magnitude of the accelerometer into a queue."""
	acc_event = types.SimpleNamespace()
	acc_event.acc_mag = (acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5
	acc_event.timestamp = timestamp - offset
	acc_event.identifier = identifier
	acc_queue.put(acc_event)

def _log_termination(func):
	"""Try to execute a function and inform about its termination by catching all exceptions."""
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		try:
			func(*args, **kwargs)
		except KeyboardInterrupt:
			LOG.debug("SIGINT termination")
		except:
			LOG.exception("unexpected termination")
		else:
			LOG.debug("normal termination")
	return wrapper

def _merge_dicts(base, new):
	"""Merge arbitrarily nested dictionaries (overwrite base with new value)."""
	for key, new_value in new.items():
		base_value = base.get(key)
		if isinstance(base_value, dict) and isinstance(new_value, dict):
			_merge_dicts(base_value, new_value)
		else:
			base[key] = new_value

def add_meta_data(prefix, new_data):
	"""Merge (and potentially overwrite) meta data."""
	file_name = f"{prefix}.meta.json"
	# read existing meta data
	try:
		with open(file_name, "r+") as meta_file:
			meta_data = json.load(meta_file)
	except (ValueError, FileNotFoundError):
		meta_data = {}
	# merge new data and write back
	_merge_dicts(meta_data, new_data)
	with open(file_name, "w") as meta_file:
		json.dump(meta_data, meta_file, indent=4, ensure_ascii=False)

@_log_termination
def log_myo(shared, prefix, identifier, init_args, record_categories, subscribe_args):
	"""Connect to and record data received from a Myo."""
	mp.current_process().name = identifier

	# define csv file headers depending on the data category
	header = {
		myo_raw.DataCategory.ARM: ["time", "arm", "direction"],
		myo_raw.DataCategory.BATTERY: ["time", "level"],
		myo_raw.DataCategory.EMG: ["time"] + _rep("emg", 8) + ["moving", "characteristics"],
		myo_raw.DataCategory.IMU: ["time"] + _rep("quat", 4) + _rep("acc", 3) + _rep("gyro", 3),
		myo_raw.DataCategory.POSE: ["time", "pose"],
	}

	# ensure that all opened files and the Myo connection are closed in any case
	with contextlib.ExitStack() as stack:
		# allow only a single process to execute the following code in order to:
		# - avoid disturbances during connection buildup
		# - avoid concurrent writing operations to the meta data file
		with shared.lock:
			# connect and disable sleep mode to prevent a disconnect
			myo = stack.enter_context(myo_raw.MyoRaw(**init_args))
			myo.set_sleep_mode(1)
			# register a callback to set the logo LED to green and the bar LED to blue on disconnect
			stack.callback(myo.set_leds, (0, 255, 0), (0, 0, 255))
			# disable the logo LED and set the bar LED to blue for indicating the connection
			myo.set_leds((0, 0, 0), (0, 0, 255))
			# store meta data
			add_meta_data(prefix, {"myo": {"devices": {identifier: myo.get_name()}}})
			# notify main thread about a successful connection buildup
			with shared.setup_condition:
				shared.setup_condition.notify()

		# await all successful connection setup of all Myos
		shared.barrier.wait()

		# create and register data handlers for all data categories to be recorded
		for category in record_categories:
			# open a csv file and write the header dependent on the categorie
			csv_file_name = f"{prefix}.{identifier}.{category.name.lower()}.csv"
			csv_file = stack.enter_context(open(csv_file_name, "w"))
			csv_writer = csv.writer(csv_file, delimiter=",")
			csv_writer.writerow(header[category])
			# ensure that the data handler is removed before closing the corresponding file
			stack.callback(myo.clear_handler, category)
			myo.add_handler(category, functools.partial(_write_data, csv_writer=csv_writer))

		# add an IMU handler to send a notification on "high" (above the threshold) acceleration
		detect_fun = functools.partial(_record_acc_mag, acc_queue=shared.acc_queue, identifier=identifier)
		functools.update_wrapper(detect_fun, _record_acc_mag)
		myo.add_handler(myo_raw.DataCategory.IMU, detect_fun)
		# start a daemon thread to remove the special IMU handler on receiving the stop_sync_flag
		def remove_sync_handler():
			shared.stop_sync_flag.wait()
			LOG.debug("popped the %s function", myo.pop_handler(myo_raw.DataCategory.IMU).__name__)
		threading.Thread(target=remove_sync_handler, daemon=True).start()

		# fix to avoid a random 0.5 s delay in the second process
		time.sleep(1)

		# synchronize all processes to simultaneously start data recording
		shared.barrier.wait()
		myo.subscribe(**subscribe_args)
		while not shared.stop_flag.is_set():
			myo.run(1)

@_log_termination
def log_keyboard(shared, prefix, escape_key, record_keys):
	"""Record keyboard events."""
	mp.current_process().name = "key"
	with contextlib.ExitStack() as stack:
		keyboard = xkeyboard.XKeyboard()

		# await all successful connection setup of all Myos
		shared.barrier.wait()

		# open file and register data handler function
		handler = None
		if record_keys:
			csv_file = stack.enter_context(open(f"{prefix}.key.csv", "w"))
			csv_writer = csv.writer(csv_file, delimiter=",")
			csv_writer.writerow(["time", "keycode", "keysym", "event"])
			handler = functools.partial(_write_data, csv_writer=csv_writer)

		# synchronize all processes to simultaneously start data recording
		shared.barrier.wait()
		keyboard.record(handler, escape_key, shared.stop_flag)

def create_tasks(prefix, escape_key, record_keys, myo_devices, subscribe_args):
	"""Create tasks and initializer arguments to record data of multiple Myos and the keyboard."""
	# derive which categories will be recorded based on the subscribe parameters
	cat_to_cond = {
		myo_raw.DataCategory.ARM: subscribe_args["clf_state"] == myo_raw.CLFState.ACTIVE,
		myo_raw.DataCategory.BATTERY: subscribe_args["battery"] == True,
		myo_raw.DataCategory.EMG: subscribe_args["emg_mode"] != myo_raw.EMGMode.OFF,
		myo_raw.DataCategory.IMU: subscribe_args["imu_mode"] != myo_raw.IMUMode.OFF,
		myo_raw.DataCategory.POSE: subscribe_args["clf_state"] == myo_raw.CLFState.ACTIVE,
	}
	record_categories = [category for category, condition in cat_to_cond.items() if condition]

	# assemble and return myo and keyboard tasks
	tasks = []
	for tty, mac, identifier in myo_devices:
		init_args = {"tty": tty, "mac": mac, "native": False}
		args = (prefix, identifier, init_args, record_categories, subscribe_args)
		tasks.append((log_myo, args))
	tasks.append((log_keyboard, (prefix, escape_key, record_keys)))
	return tasks

def create_shared_ressources(num_tasks):
	"""Create shared resources to be accessed from each thread."""
	shared = types.SimpleNamespace()
	shared.lock = mp.Lock()
	shared.barrier = mp.Barrier(num_tasks)
	shared.stop_flag = mp.Event()
	shared.stop_sync_flag = mp.Event()
	shared.setup_condition = mp.Condition()
	shared.acc_queue = mp.Queue()
	return shared

def count_unique_basename(prefix):
	"""Count unique basename given a prefix."""
	unique = set(path.name.split(".")[0] for path in pathlib.Path(".").glob(prefix + "-*"))
	return len(unique)

def enumerate_prefix(prefix):
	"""Create required parent directories and add numbering to the prefix."""
	pathlib.Path(prefix).parent.mkdir(parents=True, exist_ok=True)
	return prefix + "-" + str(count_unique_basename(prefix))

def wait_for_sync_acc(prefix, max_lag, min_acc_mag, myo_names, shared):
	"""Wait for "high" acceleration events of all Myos separated by the given time lag in s."""
	LOG.debug("waiting for high acceleration from all Myos...")
	LOG.debug("... with a lag of <= %s s", max_lag)
	LOG.debug("... with a magnitude > %.1f g", min_acc_mag / 2048)

	candidates = {name: collections.deque() for name in myo_names}
	last_timestamps = {}
	while not shared.stop_flag.is_set():
		# try to get a new accelerometer magnitude value
		try:
			new_event = shared.acc_queue.get(timeout=0.25)
		except queue.Empty:
			continue
		# store the timestamp of the last sample
		last_timestamps[new_event.identifier] = new_event.timestamp
		# filter for candidates (acc magnitude larger than the threshold)
		if new_event.acc_mag > min_acc_mag:
			LOG.debug("detected (%.1f g) from %s", new_event.acc_mag / 2048, new_event.identifier)
			# append candidates to their corresponding deques
			candidates[new_event.identifier].append(new_event.timestamp)

			# remove candidates with timestamp not compatible with:
			#    max(min(current times of all Myos), oldest candidate of all Myos)
			min_time_current = min(last_timestamps.values())
			min_time = max(candidate_list[0] if candidate_list else min_time_current
					for candidate_list in candidates.values())
			for candidate_list in candidates.values():
				while candidate_list and candidate_list[0] + max_lag < min_time:
					candidate_list.popleft()

			# check for successful clap sync
			if all(candidates.values()):
				shared.stop_sync_flag.set()
				sync_time = max(last_timestamps.values())
				LOG.debug("detected high acceleration from all Myos %.1f s after start", sync_time)
				add_meta_data(prefix, {"sync": {
					"method": "joint high acceleration event",
					"min_acc_magnitude": min_acc_mag,
					"max_lag": max_lag,
					"time": sync_time
				}})
				return

def default_task(prefix, myo_names, shared):
	"""Default example task: 1. wait for synchronous acceleration, 2. show elapsed time."""
	try:
		LOG.info(f"default task with prefix = {prefix} started")
		wait_for_sync_acc(prefix, SYNC_MAX_LAG, SYNC_MIN_ACC_MAG, myo_names, shared)
		start_time = time.time()
		while not shared.stop_flag.is_set():
			time.sleep(0.1)
			print(f"run time: {time.time() - start_time:.2f} s", end="\r")
	except KeyboardInterrupt:
		print()

def run(prefix, escape_key, record_keys, myo_devices, subscribe_args, meta, main_task=default_task):
	"""Record data of multiple Myos and of the keyboard."""
	# store common and task related configuration details in the meta data file
	add_meta_data(prefix, {"common": {"tag": meta.tag, "power_line_frequency": meta.pl_freq}})
	if record_keys:
		add_meta_data(prefix, {"keyboard": {"model": meta.kb_model, "layout": meta.kb_layout}})
	if myo_devices:
		add_meta_data(prefix, {"myo": subscribe_args})
		if len(myo_devices) > 1:
			add_meta_data(prefix, {"sync": {}})

	# create child tasks and shared ressources
	child_tasks = create_tasks(prefix, escape_key, record_keys, myo_devices, subscribe_args)
	shared = create_shared_ressources(len(child_tasks) + 1)

	while True:
		jobs, result = [], None
		shared.stop_flag.clear()
		try:
			# start child processes
			for func, args in child_tasks:
				job = mp.Process(target=func, args=(shared,) + args)
				job.start()
				jobs.append(job)

			# interrupt execution with SIGINT if any Myo does not connect within TIMEOUT seconds
			with shared.setup_condition:
				if not all(shared.setup_condition.wait(TIMEOUT) for i in range(len(myo_devices))):
					for job in jobs:
						os.kill(job.pid, signal.SIGINT)
					raise TimeoutError
			shared.barrier.wait()
			LOG.debug("all Myos successfully connected")

			# run main task when all processes start to record data
			shared.barrier.wait()
			LOG.debug("starting data recording")
			result = main_task(prefix, (dev[2] for dev in myo_devices), shared)
		except TimeoutError:
			LOG.info(f"At least one Myo did not connect after {TIMEOUT} s.")
			continue
		else:
			break
		finally:
			# whatever happend, try to shut down the child processes gracefully
			shared.stop_flag.set()
			# give the child processes some time to terminate themselves (thus avoiding a race
			# condition and also avoid a deadlock, e.g. because of lost Bluetooth packets)
			for job in jobs:
				job.join(TIMEOUT)
				job.terminate()

	# clear input buffers of sys.stdin
	termios.tcflush(sys.stdin, termios.TCIFLUSH)
	return result

def deep_sleep(myo_devices):
	"""Connect serially to every Myo and put them into deep sleep"""
	for tty, mac, identifier in myo_devices:
		LOG.info(f"putting {identifier} Myo into deep sleep state")
		with myo_raw.MyoRaw(tty=tty, mac=mac, native=False) as myo:
			myo.deep_sleep()

def extract_action(data):
	"""Create an argparser action to retrieve values from a dictionary."""
	class Action(argparse.Action):
		def __call__(self, parser, args, values, option_string=None):
			setattr(args, self.dest, data[values])
	return Action

def _main():
	"""Parse inputs and execute the run() function."""
	# define the common parser
	common_parser = argparse.ArgumentParser(add_help=False)
	common_parser.add_argument("prefix", help="the file prefix")
	common_parser.add_argument("--escape_key", metavar="KEY", default="Escape", type=str,
		help="the KEY to stop recording (default: %(default)s)")
	common_parser.add_argument("--tag", metavar="STRING", help="the description of the recorded data")
	common_parser.add_argument("--pl_freq", type=float, default=50.0, help="the power line frequency in Hz")
	common_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity")
	common_parser.add_argument("--log_file", help="the file to log to")

	# define the keyboard parser
	kb_parser = argparse.ArgumentParser(add_help=False)
	kb_parser.add_argument("--kb_model", choices=KB_CONFIG["models"].keys(), action=extract_action(KB_CONFIG["models"]),
			help="description of the physical keyboard")
	kb_parser.add_argument("--kb_layout", choices=KB_CONFIG["layouts"], help="the current keyboard layout")
	kb_default_values = vars(kb_parser.parse_args([]))

	# define the myo parser
	myo_parser = argparse.ArgumentParser(add_help=False)
	myo_parser.add_argument("--device", action="append", metavar=("TTY", "MAC", "ID"), nargs=3, default=[],
			help="the Myo dongle TTY, the Myo MAC address and a custom identifier to distinguish Myos (e.g. left/right)")
	emg_modes = ", ".join([str(item.value) + ": " + item.name for item in myo_raw.EMGMode])
	myo_parser.add_argument("--emg_mode", type=int, default=myo_raw.EMGMode.RAW, choices=list(map(int, myo_raw.EMGMode)),
			help="the EMG data recording mode ({0} - default: %(default)s)".format(emg_modes))
	imu_modes = ", ".join([str(item.value) + ": " + item.name for item in myo_raw.IMUMode])
	myo_parser.add_argument("--imu_mode", type=int, default=myo_raw.IMUMode.ON, choices=list(map(int, myo_raw.IMUMode)), metavar="{0,...,4}",
			help="the IMU data recording mode ({0} - default: %(default)s)".format(imu_modes))
	clf_states = ", ".join([str(item.value) + ": " + item.name for item in myo_raw.CLFState])
	myo_parser.add_argument("--clf_state", type=int, default=myo_raw.CLFState.PASSIVE, choices=list(map(int, myo_raw.CLFState)),
		help="the state on how to handle the on-board classifier data ({0} - default: %(default)s)".format(clf_states))
	myo_parser.add_argument("--battery", action="store_true", help="record battery data")
	myo_parser.add_argument("--deep_sleep", action="store_true", help="put all Myos into the deep sleep state")
	myo_default_values = vars(myo_parser.parse_args([]))

	# define the main parser
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="rec_mode")
	subparsers.required = True
	subparsers.add_parser("rec_none", parents=[common_parser],
			help="record nothing")
	subparsers.add_parser("rec_keyboard", parents=[common_parser, kb_parser],
			help="record keystrokes")
	subparsers.add_parser("rec_myo", parents=[common_parser, myo_parser],
			help="record sensor data from an arbitrary number of Myos")
	subparsers.add_parser("rec_all", parents=[common_parser, kb_parser, myo_parser],
			help="record keystrokes and sensor data from an arbitrary number of Myos")
	# define defaults accross subparsers
	parser.set_defaults(**kb_default_values, **myo_default_values)
	args = parser.parse_args()

	# define log level
	log_format = "[%(levelname)s/%(processName)s] %(message)s"
	log_level = max(3 - args.verbose, 0) * 10
	logging.basicConfig(filename=args.log_file, level=log_level, format=log_format)

	# derive arguments and start recording data
	meta = types.SimpleNamespace()
	meta.kb_model = args.kb_model
	meta.kb_layout = args.kb_layout
	meta.tag = args.tag
	meta.pl_freq = args.pl_freq
	subscribe_args = {k: vars(args)[k] for k in ("emg_mode", "imu_mode", "clf_state", "battery")}
	record_keys = args.rec_mode in ("rec_keyboard", "rec_all")
	prefix = enumerate_prefix(args.prefix)
	if args.deep_sleep:
		deep_sleep(args.device)
	else:
		run(prefix, args.escape_key, record_keys, args.device, subscribe_args, meta)

if __name__ == "__main__":
	_main()
