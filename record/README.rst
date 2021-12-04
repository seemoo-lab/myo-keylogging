record
======

This folder contains scripts to run our data study and post-process the resulting meta data.
The final data set created with this source code is available under: https://doi.org/10.5281/zenodo.5594651

Note that the scripts in this folder are independent from the other parts of the project and should be executed from within this folder.

Main Recording Scripts
----------------------

The three files below are the main scripts for running our data study.

- ``tasks.py``: Part one of the data study for creating the training data (textual task).
- ``passwords.py``: Part two of the data study for creating the test data (password tasks)
- ``measure.py``: Sample recording script for testing the setup and clap synchronization mechanism.

All non-random texts and passwords for the training and test data creation are located inside the ``english`` and ``german`` folders.

Utility and Definitions
-----------------------

The files below are required by the main recording scripts.

- ``record.py``: The main backbone for recording sensor and keyboard data while executing a certain task payload.
- ``utils.py``: Utility functions used for printing tasks.
- ``english_keymap.py``: Definition of keys used on an English keyboard layout, along with prompts displayed during the data study for English speaking participants.
- ``german_keymap.py``: Definition of keys used on an German keyboard layout, along with prompts displayed during the data study for German speaking participants.
- ``seed``: Our seed file for ``pwgen``.

Post-process Recorded Data
--------------------------

The four scripts below are used to fix errors occurred during the recording, as well as for updating the meta data with missing information.

- ``merge_task.py``: Script for merging files from two individual recordings (adding an artificial 30 second pause in between) in case of errors during the recording.
- ``update_text_meta.py``: Update the meta-data with additional information not stored during the original data study.
- ``update_pw_meta.py``: Update the meta-data of the test data with additional information not stored during the original data study.
- ``update_cuts.py``: Update the meta-data of the test data with a more precise list of timestamps for retrieving individual passwords attempts.

For future data studies it is advised to incorporate the three meta data update scripts into their respective recording script.

Acknowledgments
---------------

Special thanks goes to Wikipedia, the EFF and the SecLists by Daniel Miessler, Jason Haddix and g0tmi1k, for providing us with approachable texts and password candidates for our data study:

1. The file contents of the ``fun_facts``, ``minimal`` and ``pangrams`` files inside the ``english`` and ``german`` folders are taken from Wikipedia and are available under the `CC-BY-SA 3.0 license`_.
2. The file ``english/eff_long`` by the EFF is taken from `EFF's New Wordlists for Random Passphrases`_  and is available under the `CC-BY 3.0 license`_.
3. The file ``english/seclists_passwords`` by Daniel Miessler, Jason Haddix, and g0tmi1k is taken from `Top 1000 most common passwords`_ and is available under the MIT license.

.. _CC-BY-SA 3.0 license: https://creativecommons.org/licenses/by-sa/3.0/
.. _CC-BY 3.0 license: https://creativecommons.org/licenses/by/3.0/us/
.. _EFF's New Wordlists for Random Passphrases: https://www.eff.org/deeplinks/2016/07/new-wordlists-random-passphrases
.. _Top 1000 most common passwords: https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/10-million-password-list-top-1000.txt

License
-------

This source code is available under the GPLv3 license.
