Myo Keylogging
==============

This is the source code for our paper *My(o) Armband Leaks Passwords: An EMG and IMU Based Keylogging Side-Channel Attack* by Matthias Gazzari, Annemarie Mattmann, Max Maass and Matthias Hollick in Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, Volume 5, Issue 4, 2021.

We include the software used for recording the dataset (``record`` folder) and the software for training and running the neural networks (``ml`` folder) as well as analyzing the results (``analysis`` folder).
The ``scripts`` folder provides some helper scripts for automating batches of hyperparameter optimization, model fitting, analyses and more.
The ``results`` folder includes a pickled version of the predictions of our models, on which analyses can be run, e.g. to reproduce the paper results.

Installation
------------

To install the project, first clone the repository and change directory into the fresh clone:

.. code-block::

	git clone https://github.com/seemoo-lab/myo-keylogging.git
	cd myo-keylogging

You can use a python virtual environment (or any other virtual environment of your choice):

.. code-block::

	mkvirtualenv myo --system-site-packages
	workon myo

To make sure you have the newest software versions you can run an upgrade:

.. code-block::

	pip install --upgrade pip setuptools

To install the requirements run:

.. code-block::

	pip install -r requirements.txt

Finally, import the training and test data into the project.
The top level folder should include a folder ``train-data`` with all the records for training the models and a folder ``test-data`` with all the records for testing the models.

.. code-block::

	wget https://zenodo.org/record/5594651/files/myo-keylogging-dataset.zip
	unzip myo-keylogging-dataset.zip

Using the record library, you can add you can extend this dataset.

Rerun of Results
----------------

To reproduce our results from the provided predictions of our models, go to the top level directory and run:

.. code-block::

	./scripts/create_results.sh

This will recreate all performance value files and plots in the subfolders of the ``results`` folder as used in the paper.

Run the following to list the fastest and slowest typists in order to determine their class imbalance in the ``results/train-data-skew.csv`` and the ``results/test-data-skew.csv`` files:

.. code-block::

	python -m analysis exp_key_data

To recreate the provided predictions and class skew files, execute the following from the top level directory:

.. code-block::

	./scripts/create_models.sh
	./scripts/create_predictions.sh
	./scripts/create_class_skew_files.sh

This will fit the models with the current choice of hyperparameters and run each model on the test dataset to create the required predictions for analysis.
Additionally, the class skew files will be recreated.

To run the hyperparameter optimization either run the ``run_shallow_hpo.sh`` script or, alternatively, the ``slurm_run_shallow_hpo.sh`` script when on a SLURM cluster.

.. code-block::

	sbatch scripts/slurm_run_shallow_hpo.sh
	./scripts/run_shallow_hpo.sh

Afterwards you can use the ``merge_shallow_hpo_runs.py`` script to combine the results for easier evaluation of the hyperparameters.

Fit Models
----------

In order to fit and analyze your own models, go to the top level directory and run any of:

.. code-block::

	python -m ml crnn
	python -m ml resnet
	python -m ml resnet11
	python -m ml wavenet

This will fit the respective model with the default parameters and in binary mode for keystroke detection.
In order to fit multiclass models for keystroke identification, use the ``encoding`` parameter, e.g.:

.. code-block::

	python -m ml crnn --encoding "multiclass"

In order to test specific sensors, ignore the others (note that quaternions are ignored by default), e.g. to use only EMG on a CRNN model, use:

.. code-block::

	python -m ml crnn --ignore "quat" "acc" "gyro"

To run a hyperparameter optimization, run e.g.:

.. code-block::

	python -m ml crnn --func shallow_hpo --step 5

To gain more information on possible parameters, run e.g.:

.. code-block::

	python -m ml crnn --help

Some parameters for the neural networks are fixed in the code.

Analyze Models
--------------

In order to analyze your models, run ``apply_models`` to create the predictions as pickled files. On these you can run further analyses found in the ``analysis`` folder.

To run ``apply_models`` on a binary model, do:

.. code-block::

	python -m analysis apply_models --model_path results/<PATH_TO_MODEL> --encoding binary --data_path test-data/ --save_path results/<PATH_TO_PKL> --save_only --basenames <YOUR MODELS>

To run a multiclass model, do:

.. code-block::

	python -m analysis apply_models --model_path results/<PATH_TO_MODEL> --encoding multiclass --data_path test-data/ --save_path results/<PATH_TO_PKL> --save_only --basenames <YOUR MODELS>

To chain a binary and multiclass model, do e.g.:

.. code-block::

	python -m analysis apply_models --model_path results/<PATH_TO_MODEL> --encoding chain --data_path test-data/ --save_path results/<PATH_TO_PKL> --save_only --basenames <YOUR MODELS> --tolerance 10

Further parameters interesting for analyses may be a filter on the users with the parameter (``--users known`` or ``--users unknown``) or on the data (``--data known`` or ``--data unknown``) to include only users (not) in the training data or include only data typed by all or no other user respectively.

For more information, run:

.. code-block::

	python -m analysis apply_models --help

To later recreate model performance results and plots, run:

.. code-block::

	python -m analysis apply_models --encoding <ENCODING> --load_results results/<PATH_TO_PKL> --save_path results/<PATH_TO_PKL> --save_only

with the appropriate encoding of the model used to create the pickled results.

To run further analyses on the generated predictions, create or choose your analysis from the ``analysis`` folder and run:

.. code-block::

	python -m analysis <ANALYSIS_NAME>

Refer to the help for further information:

.. code-block::

	python -m analysis <ANALYSIS_NAME> --help

Record Data
-----------

In order to record your own data(set), switch to the record folder.
To record sensor data with our recording software, you will need one to two Myo armbands connected to your computer.
Then, you can start a training data recording, e.g.:

.. code-block::

	python tasks.py -s 42 -l german record touch_typing --left_tty <TTY_LEFT_MYO> --left_mac <MAC_LEFT_MYO> --right_tty <TTY_RIGHT_MYO> --right_mac <MAC_RIGHT_MYO> --kb_model TADA68_DE

for a German recording with seed 42, a touch typist and a TADA68 German physical keyboard layout or

.. code-block::

	python tasks.py -s 42 -l english record touch_typing --left_tty <TTY_LEFT_MYO> --left_mac <MAC_LEFT_MYO> --right_tty <TTY_RIGHT_MYO> --right_mac <MAC_RIGHT_MYO> --kb_model TADA68_US

for an English recording with seed 42, a hybrid typist and a TADA68 English physical keyboard layout.

In order to start a test data recording, simply run the
``passwords.py`` instead of the ``tasks.py``.

After recording training data, please execute the following script to complete the meta data:

.. code-block::

		python update_text_meta.py -p ../train-data/

After recording test data, please execute the following two scripts to complete the meta data:

.. code-block::

		python update_pw_meta.py -p ../test-data/
		python update_cuts.py -p ../test-data/

For further information, check:

.. code-block::

	python tasks.py --help
	python passwords.py --help

Note that the recording software includes text extracts as outlined in the acknowledgments below.

Links
-----

- Our paper: https://doi.org/10.1145/3494986 (on arXiv: https://arxiv.org/abs/2112.02382)
- Our dataset: https://doi.org/10.5281/zenodo.5594651

Acknowledgments
---------------

This work includes the following external materials to be found in the ``record`` folder:

1. Various texts from Wikipedia available under the `CC-BY-SA 3.0 license`_.
2. The `EFF's New Wordlists for Random Passphrases`_  available under the `CC-BY 3.0 license`_.
3. An extract of the `Top 1000 most common passwords`_ by Daniel Miessler, Jason Haddix, and g0tmi1k available under the MIT license.

.. _CC-BY-SA 3.0 license: https://creativecommons.org/licenses/by-sa/3.0/
.. _CC-BY 3.0 license: https://creativecommons.org/licenses/by/3.0/us/
.. _EFF's New Wordlists for Random Passphrases: https://www.eff.org/deeplinks/2016/07/new-wordlists-random-passphrases
.. _Top 1000 most common passwords: https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/10-million-password-list-top-1000.txt

License
-------

This software is licensed under the GPLv3 license, please also refer to the LICENSE file.
