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

import argparse
import pathlib
import importlib
import inspect
import sys
import logging

LOG = logging.getLogger(__name__)

def load_executable_module(pkg_name, module_name):
	"""
	Try to load a module and return None if that fails or if it does not contain a main function.
	"""
	try:
		module = importlib.import_module(pkg_name + "." + module_name)
	except ModuleNotFoundError as error:
		LOG.warning("Cannot import %s (%s)", module_name, error)
		return None

	# skip modules without a main function
	if not hasattr(module, "main"):
		LOG.info("skipping %s (has no main function)", module)
		return None

	return module

def execute(pkg_file, pkg_name):
	"""
	Create an argparser with subparsers for each module with a main() function inside the package
	and execute the requested main function. Positional and optional parameters are automatically
	determined as defined by the main function.
	Additionally, default values and annotations are displayed.
	"""
	try:
		parser = argparse.ArgumentParser(prog="python -m " + pkg_name)
		subparsers = parser.add_subparsers(dest="'any module inside %s'" % pkg_name, required=True)
	except TypeError:
		# Python < 3.7 fix (c.f. https://bugs.python.org/issue26510 for more details)
		parser = argparse.ArgumentParser(prog="python -m " + pkg_name)
		subparsers = parser.add_subparsers(dest="'any module inside %s'" % pkg_name)
		subparsers.required = True

	# get all available modules from the requested package
	available_module_names = []
	for path in pathlib.Path(pkg_file).parent.glob("*.py"):
		if path.is_file() and not path.stem.startswith("_"):
			available_module_names.append(path.stem)

	# load a specified module or all modules if the specified one does not exist/is not loadable
	chosen_modules = {}
	if len(sys.argv) > 1 and sys.argv[1] in available_module_names:
		module = load_executable_module(pkg_name, sys.argv[1])
		if module:
			chosen_modules[sys.argv[1]] = module
	if not chosen_modules:
		for module_name in available_module_names:
			module = load_executable_module(pkg_name, module_name)
			if module:
				chosen_modules[module_name] = module

	# generate subparsers for the chosen modules
	for module_name, module in chosen_modules.items():
		# create a subparser for the main function of the imported module
		subparser = subparsers.add_parser(module_name, description=module.__doc__, help=module.__doc__)
		subparser.set_defaults(_func=module.main)

		# create positional and optional arguments for the main function
		signature = inspect.signature(module.main)
		for param in signature.parameters.values():
			annotation = "" if param.annotation is inspect.Parameter.empty else param.annotation
			if param.default is inspect.Parameter.empty:
				argument = param.name
				default = None
				help_msg = annotation
			else:
				argument = "--" + param.name
				default = param.default
				help_msg = annotation + " (default: %(default)s)"
			if isinstance(param.default, bool):
				action = "store_false" if default else "store_true"
				subparser.add_argument(argument, action=action, help=help_msg)
			elif not isinstance(param.default, str) and hasattr(param.default, "__len__"):
				arg_type = type(param.default[0]) if len(param.default) > 0 else str
				subparser.add_argument(argument, default=default, nargs='*', help=help_msg)
			else:
				arg_type = type(default) if default is not None else str
				subparser.add_argument(argument, default=default, type=arg_type, help=help_msg)
	args = parser.parse_args()

	# call the requested function by passing arguments in the defined order
	cmd_signature = inspect.signature(args._func)
	params = [vars(args)[param.name] for param in cmd_signature.parameters.values()]
	args._func(*params)
