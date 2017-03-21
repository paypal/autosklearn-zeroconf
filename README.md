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

## Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.

### Install auto-sklearn
<code>
# On Ubuntu
wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -O Anaconda3-Linux-x86_64.sh
chmod u+x Anaconda3-Linux-x86_64.sh
./Anaconda3-Linux-x86_64.sh
# A compiler is needed to compile a few things the from requirements.txt
# Chose for your Linux flavor
# On Ubuntu
sudo apt-get install gcc build-essential 
# On RedHat
yum -y groupinstall 'Development Tools'
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
pip install auto-sklearn
</code>

## License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

