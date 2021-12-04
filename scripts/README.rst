Scripts
=======

This folder contains scripts to automate certain parts of our project.
Note that every script is designed to be executed from the top level directory of this repository.

Calculate Class Skew
--------------------

Run the ``create_class_skew_files.sh`` script to create an extensive overview of the existing class skew in both datasets.

Hyperparameter Optimization
---------------------------

To run the hyperparameter optimization either run the ``run_shallow_hpo.sh`` script or, alternatively, the ``slurm_run_shallow_hpo.sh`` script when on a SLURM cluster.
Afterwards you can use the ``merge_shallow_hpo_runs.py`` script to combine the results for easier evaluation of the hyperparameters.

Create Models and Prediction Pickles
------------------------------------

To train the neural networks just execute the ``create_models.sh`` script.
It will generate different models (binary, multi class, trained on different subset of the sensor modalities) for each available neural network architecture.
The resulting models can then be used to recreate the final predictions (stored in pickled files) with the ``create_predictions.sh`` script.
Note that the result of this run is already available in this repository.

Recreate Paper Results
----------------------

To recreate the results of our paper simply execute the ``create_results.sh`` script to use the provided or newly generated prediction pickles to create all plots, tables and results from our paper.
