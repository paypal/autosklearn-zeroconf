## What is autosklearn-zeroconf
The autosklearn-zeroconf file takes a dataframe of any size and trains https://github.com/automl/auto-sklearn binary classifier ensemble. No configuration is needed as the name suggests.
As a result running auto-sklearn becomes a "fire and forget" type of operation. It greatly increases the utility and decreases turnaround time for experiments.

## How Does It Work
To keep the training time reasonable autosklearn-zeroconf samples the data and tests all the models from autosklearn library on it once. The results of the test (duration) is used to calculate the per_run_time_limit, time_left_for_this_task and number of seeds parameters for autosklearn. The code also converts the panda dataframe into a form that autosklearn can handle (categorical and float datatypes).

## Running autosklearn-zeroconf
To run autosklearn-zeroconf start '''zeroconf.py your_dataframe.h5''' from command line.

## Data Format
The code uses a pandas dataframe format to manage the data. It is stored in the HDF5 file for convenience.
As an example you can run autosklearn-zeroconf on a widely known Titanic dataset from Kaggle https://www.kaggle.com/c/titanic/data .
Download these two csv files https://www.kaggle.com/c/titanic/download/train.csv https://www.kaggle.com/c/titanic/download/test.csv and use 
zeroconf-load-dataset-Titanic.py convert them into one HDF5 file Titanic.h5

## License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

