The zeroconf.py file takes a pandas dataframe of any size and trains autosklearn binary classifier ensemble. 

It does so in a reasonable time sampling the data and testing all the models from autosklearn library on it. The results of the test (duration) is used to calculate the per_run_time_limit, time_left_for_this_task and number of seeds parameters for autosklearn. The code also converts the panda dataframe into a form that autosklearn can handle (categorical and float datatypes).

As a result running autosklearn becomes "fire and forget" type of operation. It greatly increases the utility and decreases turnaround time for experiments.

Use Titanic.h5 as an example dataset. It is in the form of a pandas dataframe stored in hdf5 store. It has test and train data in one file.

zeroconf-load-dataset-Titanic.py serves as an example on how to convert a CSV file into HDF5 like Titanic.h5


##License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

