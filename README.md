## What is autosklearn-zeroconf
The autosklearn-zeroconf file takes a dataframe of any size and trains [auto-sklearn](https://github.com/automl/auto-sklearn) binary classifier ensemble. No configuration is needed as the name suggests. Auto-sklearn is the recent [AutoML Challenge](http://www.kdnuggets.com/2016/08/winning-automl-challenge-auto-sklearn.html) winner.

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
 sgd

plus samplers, scalers, imputers (14 feature processing methods, and 3 data preprocessing
methods,  giving  rise  to  a  structured  hypothesis  space  with  100+  hyperparameters)

## Running autosklearn-zeroconf
To run autosklearn-zeroconf start '''python zeroconf.py your_dataframe.h5 2>/dev/null|grep ZEROCONF''' from command line.
The script was tested on Ubuntu and RedHat. It won't work on any WindowsOS because auto-sklearn doesn't support Windows.

## Data Format
The code uses a pandas dataframe format to manage the data. It is stored in the HDF5 file for convenience.
As an example you can run autosklearn-zeroconf on a widely known Titanic dataset from Kaggle https://www.kaggle.com/c/titanic/data .
Download these two csv files https://www.kaggle.com/c/titanic/download/train.csv https://www.kaggle.com/c/titanic/download/test.csv and use 
zeroconf-load-dataset-Titanic.py convert them into one HDF5 file Titanic.h5

## Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.
Alternatively you could use git clone
<pre>
sudo apt-get update && sudo apt-get install git && git clone https://github.com/paypal/autosklearn-zeroconf.git
</pre>

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

## Example of the output
<pre>
python zeroconf.py Titanic.h5 2>/dev/null|grep ZEROCONF
[ZEROCONF] Read dataset from the store # 20:14 #
[ZEROCONF] Values of y [ 0  1 -1] # 20:14 #
[ZEROCONF] Filling missing values in X with the most frequent values # 20:14 #
[ZEROCONF] Factorizing the X # 20:14 #
[ZEROCONF] Dataframe split into X and y # 20:14 #
[ZEROCONF] Preparing a sample to measure approx classifier run time and select features # 20:14 #
[ZEROCONF] Reserved 33% of the training dataset for validation (upto 33k rows) # 20:14 #
[ZEROCONF] Constructing preprocessor pipeline and transforming sample data # 20:14 #
[ZEROCONF] Running estimators on the sample # 20:14 #
[ZEROCONF] decision_tree starting # 20:14 #
[ZEROCONF] decision_tree training time: 0.0013499259948730469 # 20:14 #
[ZEROCONF] gaussian_nb starting # 20:14 #
[ZEROCONF] gaussian_nb training time: 0.03557109832763672 # 20:14 #
[ZEROCONF] bernoulli_nb starting # 20:14 #
[ZEROCONF] bernoulli_nb training time: 0.05591177940368652 # 20:14 #
[ZEROCONF] k_nearest_neighbors starting # 20:14 #
[ZEROCONF] k_nearest_neighbors training time: 0.03613591194152832 # 20:14 #
[ZEROCONF] liblinear_svc starting # 20:14 #
[ZEROCONF] liblinear_svc training time: 0.02362370491027832 # 20:14 #
[ZEROCONF] lda starting # 20:14 #
[ZEROCONF] lda training time: 0.12917685508728027 # 20:14 #
[ZEROCONF] multinomial_nb starting # 20:14 #
[ZEROCONF] multinomial_nb training time: 0.06360983848571777 # 20:14 #
[ZEROCONF] passive_aggressive starting # 20:14 #
[ZEROCONF] passive_aggressive training time: 0.10781621932983398 # 20:14 #
[ZEROCONF] sgd starting # 20:14 #
[ZEROCONF] sgd training time: 0.0836634635925293 # 20:14 #
[ZEROCONF] adaboost starting # 20:14 #
[ZEROCONF] adaboost training time: 0.4440269470214844 # 20:14 #
[ZEROCONF] extra_trees starting # 20:14 #
[ZEROCONF] extra_trees training time: 0.7000019550323486 # 20:14 #
[ZEROCONF] gradient_boosting starting # 20:14 #
[ZEROCONF] gradient_boosting training time: 0.7240493297576904 # 20:14 #
[ZEROCONF] random_forest starting # 20:14 #
[ZEROCONF] random_forest training time: 0.6279892921447754 # 20:14 #
[ZEROCONF] Test classifier fit completed # 20:14 #
[ZEROCONF] per_run_time_limit=3 # 20:14 #
[ZEROCONF] Process pool size=1 # 20:14 #
[ZEROCONF] Starting autosklearn classifiers fiting # 20:14 #
[ZEROCONF] Max time allowance for a model 1 minute(s) # 20:14 #
[ZEROCONF] Overal run time is about 10 minute(s) # 20:14 #
[ZEROCONF] Starting seed=2 # 20:14 #
[ZEROCONF] ####### Finished seed=2 # 20:19 #
[ZEROCONF] Multicore fit completed # 20:19 #
[ZEROCONF] Building ensemble # 20:19 #
[ZEROCONF] Ensemble built # 20:20 #
[ZEROCONF] Show models # 20:20 #
[ZEROCONF] [(0.400000, SimpleClassificationPipeline({'preprocessor:liblinear_svc_preprocessor:tol': 0.00010000000000000009, 'balancing:strategy': 'weighting', 'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge', 'classifier:adaboost:algorithm': 'SAMME', 'preprocessor:liblinear_svc_preprocessor:C': 1.0, 'preprocessor:liblinear_svc_preprocessor:penalty': 'l1', 'preprocessor:__choice__': 'liblinear_svc_preprocessor', 'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True', 'one_hot_encoding:use_minimum_fraction': 'False', 'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1, 'preprocessor:liblinear_svc_preprocessor:dual': 'False', 'classifier:adaboost:learning_rate': 0.026303797714332906, 'rescaling:__choice__': 'minmax', 'classifier:adaboost:max_depth': 2, 'imputation:strategy': 'median', 'classifier:adaboost:n_estimators': 143, 'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr', 'classifier:__choice__': 'adaboost'}, # 20:20 #
[ZEROCONF] dataset_properties={ # 20:20 #
[ZEROCONF]   'task': 1, # 20:20 #
[ZEROCONF]   'target_type': 'classification', # 20:20 #
[ZEROCONF]   'sparse': False, # 20:20 #
[ZEROCONF]   'signed': False, # 20:20 #
[ZEROCONF]   'multilabel': False, # 20:20 #
[ZEROCONF]   'multiclass': False})), # 20:20 #
[ZEROCONF] (0.300000, SimpleClassificationPipeline({'classifier:random_forest:min_samples_leaf': 5, 'balancing:strategy': 'weighting', 'classifier:random_forest:bootstrap': 'False', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_features': 4.138756484748367, 'classifier:random_forest:min_samples_split': 9, 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:n_estimators': 100, 'rescaling:__choice__': 'standardize', 'preprocessor:__choice__': 'no_preprocessing', 'imputation:strategy': 'median', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:max_depth': 'None'}, # 20:20 #
[ZEROCONF] dataset_properties={ # 20:20 #
[ZEROCONF]   'task': 1, # 20:20 #
[ZEROCONF]   'target_type': 'classification', # 20:20 #
[ZEROCONF]   'sparse': False, # 20:20 #
[ZEROCONF]   'signed': False, # 20:20 #
[ZEROCONF]   'multilabel': False, # 20:20 #
[ZEROCONF]   'multiclass': False})), # 20:20 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:passive_aggressive:loss': 'squared_hinge', 'balancing:strategy': 'weighting', 'classifier:passive_aggressive:n_iter': 412, 'classifier:passive_aggressive:fit_intercept': 'True', 'classifier:__choice__': 'passive_aggressive', 'rescaling:__choice__': 'minmax', 'preprocessor:__choice__': 'no_preprocessing', 'imputation:strategy': 'mean', 'classifier:passive_aggressive:C': 0.002134508671186945}, # 20:20 #
[ZEROCONF] dataset_properties={ # 20:20 #
[ZEROCONF]   'task': 1, # 20:20 #
[ZEROCONF]   'target_type': 'classification', # 20:20 #
[ZEROCONF]   'sparse': False, # 20:20 #
[ZEROCONF]   'signed': False, # 20:20 #
[ZEROCONF]   'multilabel': False, # 20:20 #
[ZEROCONF]   'multiclass': False})), # 20:20 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:adaboost:max_depth': 2, 'balancing:strategy': 'weighting', 'classifier:adaboost:algorithm': 'SAMME.R', 'classifier:__choice__': 'adaboost', 'rescaling:__choice__': 'minmax', 'preprocessor:__choice__': 'no_preprocessing', 'imputation:strategy': 'median', 'classifier:adaboost:n_estimators': 232, 'classifier:adaboost:learning_rate': 0.10000000000000002}, # 20:20 #
[ZEROCONF] dataset_properties={ # 20:20 #
[ZEROCONF]   'task': 1, # 20:20 #
[ZEROCONF]   'target_type': 'classification', # 20:20 #
[ZEROCONF]   'sparse': False, # 20:20 #
[ZEROCONF]   'signed': False, # 20:20 #
[ZEROCONF]   'multilabel': False, # 20:20 #
[ZEROCONF]   'multiclass': False})), # 20:20 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:min_samples_leaf': 7, 'balancing:strategy': 'none', 'classifier:random_forest:bootstrap': 'True', 'classifier:__choice__': 'random_forest', 'one_hot_encoding:minimum_fraction': 0.01626967550201217, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_features': 1.6830354508687444, 'classifier:random_forest:min_samples_split': 5, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:n_estimators': 100, 'rescaling:__choice__': 'minmax', 'preprocessor:__choice__': 'no_preprocessing', 'imputation:strategy': 'median', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:max_depth': 'None'}, # 20:20 #
[ZEROCONF] dataset_properties={ # 20:20 #
[ZEROCONF]   'task': 1, # 20:20 #
[ZEROCONF]   'target_type': 'classification', # 20:20 #
[ZEROCONF]   'sparse': False, # 20:20 #
[ZEROCONF]   'signed': False, # 20:20 #
[ZEROCONF]   'multilabel': False, # 20:20 #
[ZEROCONF]   'multiclass': False})), # 20:20 #
[ZEROCONF] ] # 20:20 #
[ZEROCONF] Validating # 20:20 #
[ZEROCONF] Predicting on validation set # 20:20 #
[ZEROCONF] Accuracy score 0.803389830508 # 20:20 #
[ZEROCONF] ########################################################################
[ZEROCONF] The below scores are calculated for predicting '1' category value # 20:20 #
[ZEROCONF] Precision: 76%, Recall: 71%, F1: 0.73
[ZEROCONF] Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall
[ZEROCONF] [[157  25] # 20:20 #
[ZEROCONF]  [ 33  80]] # 20:20 #
[ZEROCONF] Baseline 113 positives from 295 overall = 38.3%
[ZEROCONF] ########################################################################
[ZEROCONF] Re-fitting the model ensemble on full known dataset to prepare for prediciton. This can take a long time. # 20:20 #
[ZEROCONF] Dataframe split into X and y # 20:20 #
[ZEROCONF] Predicting. This can take a long time for a large prediction set. # 20:20 #
[ZEROCONF] Prediction done # 20:20 #
[ZEROCONF] Exporting the data # 20:20 #
[ZEROCONF] ##### Zeroconf Script Completed! ##### # 20:20 #
</pre>

## Workarounds
these are not related to the autosklearn-zeroconf or auto-sklearn but rather general issues depending on your python and OS installation
### xgboost installation can not find libraries
search for them with <pre>sudo find / -name libgomp.so.1/usr/lib/x86_64-linux-gnu/libgomp.so.1</pre> and explicitly add them to the libraries path
<pre>export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6":"/usr/lib/x86_64-linux-gnu/libgomp.so.1"; python zeroconf.py Titanic.h5 2>/dev/null|grep ZEROCONF</pre>
