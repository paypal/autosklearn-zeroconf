## What is autosklearn-zeroconf
The autosklearn-zeroconf file takes a dataframe of any size and trains [auto-sklearn](https://github.com/automl/auto-sklearn) binary classifier ensemble. No configuration is needed as the name suggests. Auto-sklearn is the recent [AutoML Challenge](http://www.kdnuggets.com/2016/08/winning-automl-challenge-auto-sklearn.html) winner, Microsoft Research supported the organization of this challenge and donated the prizes.

As a result of using automl-zeroconf running auto-sklearn becomes a "fire and forget" type of operation. It greatly increases the utility and decreases turnaround time for experiments.

The main value proposition is that a data analyst or a data savvy business user can quickly run the iterations on the data (actual sources and feature design) side and on the ML side not a bit has to be changed. So it's a great tool for people not doing hardcore data science full time. Up to 90% of (marketing) data analysts may fall into this target group currently. 

## How Does It Work
To keep the training time reasonable autosklearn-zeroconf samples the data and tests all the models from autosklearn library on it once. The results of the test (duration) is used to calculate the per_run_time_limit, time_left_for_this_task and number of seeds parameters for autosklearn. The code also converts the panda dataframe into a form that autosklearn can handle (categorical and float datatypes).
<img src=https://github.com/paypal/autosklearn-zeroconf/blob/master/AutosklearnModellingLossOverTimeExample.png></img>

## Algoritms included
 bernoulli_nb,
 extra_trees,
 gaussian_nb,
 adaboost,
 gradient_boosting,
 k_nearest_neighbors,
 lda,
 liblinear_svc,
 multinomial_nb,
 passive_aggressive,
 random_forest,
 sgd,

plus samplers, scalers, imputers

## Running autosklearn-zeroconf
To run autosklearn-zeroconf start '''python zeroconf.py your_dataframe.h5''' from command line.
The script was tested on Ubuntu and RedHat. It won't work on any WindowsOS because auto-sklearn doesn't support Windows.

## Data Format
The code uses a pandas dataframe format to manage the data. It is stored in the HDF5 file for convenience.
As an example you can run autosklearn-zeroconf on a widely known Titanic dataset from Kaggle https://www.kaggle.com/c/titanic/data .
Download these two csv files https://www.kaggle.com/c/titanic/download/train.csv https://www.kaggle.com/c/titanic/download/test.csv and use 
zeroconf-load-dataset-Titanic.py convert them into one HDF5 file Titanic.h5

## Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.

### Install auto-sklearn
<pre>
# If you have no Python environment installed install Anaconda.
wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -O Anaconda3-Linux-x86_64.sh
chmod u+x Anaconda3-Linux-x86_64.sh
./Anaconda3-Linux-x86_64.sh
# A compiler is needed to compile a few things the from requirements.txt
# Chose just the line for your Linux flavor below
# On Ubuntu
sudo apt-get install gcc build-essential 
# On RedHat
yum -y groupinstall 'Development Tools'

curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
pip install auto-sklearn
</pre>

## License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

