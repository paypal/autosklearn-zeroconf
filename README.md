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

## Example
As an example you can run autosklearn-zeroconf on a "Census Income" dataset https://archive.ics.uci.edu/ml/datasets/Adult.
'''python zeroconf.py Adult.h5 2>/dev/null|grep ZEROCONF'''
And then to evaluate the prediction stored in zerconf-result.csv against the test dataset file adult.test.withid 
'''python evaluate-dataset-Adult.py'''

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
conda install libgcc
# A compiler is also needed to compile a few things the from requirements.txt
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
python zeroconf.py Adult.h5 2>&1|grep ZEROCONF
[ZEROCONF] Read dataset from the store # 00:12:53 #
[ZEROCONF] Values of y [  0.   1.  nan] # 00:12:53 #
[ZEROCONF] We need to protect NAs in y from the prediction dataset so we convert them to -1 # 00:12:53 #
[ZEROCONF] New values of y [ 0.  1. -1.] # 00:12:53 #
[ZEROCONF] Filling missing values in X with the most frequent values # 00:12:53 #
[ZEROCONF] Factorizing the X # 00:12:53 #
[ZEROCONF] Dataframe split into X and y # 00:12:53 #
[ZEROCONF] Preparing a sample to measure approx classifier run time and select features # 00:12:53 #
[ZEROCONF] Reserved 33% of the training dataset for validation (upto 33k rows) # 00:12:53 #
[ZEROCONF] Constructing preprocessor pipeline and transforming sample data # 00:12:53 #
[ZEROCONF] Running estimators on the sample # 00:12:54 #
[ZEROCONF] bernoulli_nb starting # 00:12:54 #
[ZEROCONF] bernoulli_nb training time: 0.030226469039916992 # 00:12:54 #
[ZEROCONF] gaussian_nb starting # 00:12:54 #
[ZEROCONF] gaussian_nb training time: 0.02520585060119629 # 00:12:54 #
[ZEROCONF] decision_tree starting # 00:12:54 #
[ZEROCONF] decision_tree training time: 0.02187943458557129 # 00:12:54 #
[ZEROCONF] lda starting # 00:12:55 #
[ZEROCONF] lda training time: 0.10130095481872559 # 00:12:55 #
[ZEROCONF] multinomial_nb starting # 00:12:55 #
[ZEROCONF] multinomial_nb training time: 0.11277103424072266 # 00:12:55 #
[ZEROCONF] liblinear_svc starting # 00:12:55 #
[ZEROCONF] liblinear_svc training time: 0.23301315307617188 # 00:12:55 #
[ZEROCONF] passive_aggressive starting # 00:12:55 #
[ZEROCONF] passive_aggressive training time: 0.25881266593933105 # 00:12:55 #
[ZEROCONF] sgd starting # 00:12:55 #
[ZEROCONF] sgd training time: 0.4245338439941406 # 00:12:55 #
[ZEROCONF] adaboost starting # 00:12:54 #
[ZEROCONF] adaboost training time: 1.2189135551452637 # 00:12:56 #
[ZEROCONF] k_nearest_neighbors starting # 00:12:55 #
[ZEROCONF] k_nearest_neighbors training time: 1.1744468212127686 # 00:12:56 #
[ZEROCONF] xgradient_boosting starting # 00:12:55 #
[ZEROCONF] xgradient_boosting training time: 1.6398138999938965 # 00:12:56 #
[ZEROCONF] extra_trees starting # 00:12:55 #
[ZEROCONF] extra_trees training time: 2.6967108249664307 # 00:12:57 #
[ZEROCONF] random_forest starting # 00:12:55 #
[ZEROCONF] random_forest training time: 2.7534499168395996 # 00:12:57 #
[ZEROCONF] gradient_boosting starting # 00:12:54 #
[ZEROCONF] gradient_boosting training time: 4.280401945114136 # 00:12:59 #
[ZEROCONF] Test classifier fit completed # 00:12:59 #
[ZEROCONF] per_run_time_limit=4 # 00:12:59 #
[ZEROCONF] Process pool size=2 # 00:12:59 #
[ZEROCONF] Starting autosklearn classifiers fiting on a 67% sample up to 67k rows # 00:12:59 #
[ZEROCONF] Max time allowance for a model 1 minute(s) # 00:12:59 #
[ZEROCONF] Overal run time is about 8 minute(s) # 00:12:59 #
[ZEROCONF] Starting seed=2 # 00:13:01 #
[ZEROCONF] Starting seed=3 # 00:13:02 #
[ZEROCONF] ####### Finished seed=3 # 00:16:16 #
[ZEROCONF] ####### Finished seed=2 # 00:16:18 #
[ZEROCONF] Multicore fit completed # 00:16:18 #
[ZEROCONF] Building ensemble # 00:16:18 #
[ZEROCONF] Ensemble built # 00:16:40 #
[ZEROCONF] Show models # 00:16:40 #
[ZEROCONF] [(0.200000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'entropy', 'classifier:random_forest:max_features': 0.8088696708652277, 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_samples_split': 10, 'classifier:random_forest:min_samples_leaf': 3, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'most_frequent', 'balancing:strategy': 'weighting', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'none'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.200000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'entropy', 'classifier:random_forest:max_features': 0.8088696708652277, 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_samples_split': 13, 'classifier:random_forest:min_samples_leaf': 3, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'most_frequent', 'balancing:strategy': 'weighting', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'none'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.200000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'one_hot_encoding:minimum_fraction': 0.41124722647909795, 'classifier:random_forest:max_features': 3.8870279355338146, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:min_samples_split': 17, 'classifier:random_forest:min_samples_leaf': 11, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'most_frequent', 'balancing:strategy': 'none', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'True', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'one_hot_encoding:minimum_fraction': 0.01, 'classifier:random_forest:max_features': 1.0, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:min_samples_split': 2, 'classifier:random_forest:min_samples_leaf': 1, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'mean', 'balancing:strategy': 'none', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'standardize'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'balancing:strategy': 'none', 'classifier:xgradient_boosting:min_child_weight': 10, 'classifier:xgradient_boosting:n_estimators': 342, 'classifier:xgradient_boosting:max_delta_step': 0, 'one_hot_encoding:minimum_fraction': 0.2547323861133189, 'classifier:xgradient_boosting:reg_lambda': 1, 'preprocessor:__choice__': 'no_preprocessing', 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:xgradient_boosting:max_depth': 3, 'classifier:xgradient_boosting:colsample_bylevel': 1, 'classifier:xgradient_boosting:scale_pos_weight': 1, 'classifier:xgradient_boosting:base_score': 0.5, 'classifier:xgradient_boosting:gamma': 0, 'classifier:xgradient_boosting:subsample': 0.6025298628426271, 'imputation:strategy': 'median', 'classifier:__choice__': 'xgradient_boosting', 'classifier:xgradient_boosting:learning_rate': 0.04534535307037642, 'classifier:xgradient_boosting:colsample_bytree': 1, 'classifier:xgradient_boosting:reg_alpha': 0, 'rescaling:__choice__': 'minmax'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'one_hot_encoding:minimum_fraction': 0.1664815690894424, 'classifier:random_forest:max_features': 0.8590103711882495, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:min_samples_split': 17, 'classifier:random_forest:min_samples_leaf': 1, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'median', 'balancing:strategy': 'none', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:criterion': 'gini', 'one_hot_encoding:minimum_fraction': 0.06557167597753223, 'classifier:random_forest:max_features': 4.123805727190659, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:min_samples_split': 12, 'classifier:random_forest:min_samples_leaf': 18, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'mean', 'balancing:strategy': 'none', 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax'}, # 00:16:40 #
[ZEROCONF] dataset_properties={ # 00:16:40 #
[ZEROCONF]   'target_type': 'classification', # 00:16:40 #
[ZEROCONF]   'sparse': False, # 00:16:40 #
[ZEROCONF]   'signed': False, # 00:16:40 #
[ZEROCONF]   'multilabel': False, # 00:16:40 #
[ZEROCONF]   'task': 1, # 00:16:40 #
[ZEROCONF]   'multiclass': False})), # 00:16:40 #
[ZEROCONF] ] # 00:16:40 #
[ZEROCONF] Validating # 00:16:40 #
[ZEROCONF] Predicting on validation set # 00:16:40 #
[ZEROCONF]  # 00:16:43 #
[[ZEROCONF] ######################################################################## # 00:16:43 #
[ZEROCONF] Accuracy score 86% # 00:16:43 #
[ZEROCONF] The below scores are calculated for predicting '1' category value # 00:16:43 #
[ZEROCONF] Precision: 70%, Recall: 73%, F1: 0.71 # 00:16:43 #
[ZEROCONF] Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall # 00:16:43 #
[ZEROCONF] [[7369  789] # 00:16:43 #
[ZEROCONF]  [ 710 1878]] # 00:16:43 #
[ZEROCONF] Baseline 2588 positives from 10746 overall = 24.1% # 00:16:43 #
[ZEROCONF] ######################################################################## # 00:16:43 #
[ZEROCONF]  # 00:16:43 #
[ZEROCONF] Dataframe split into X and y # 00:16:43 #
[ZEROCONF] Re-fitting the model ensemble on full known dataset to prepare for prediciton. This can take a long time. # 00:16:43 #
[ZEROCONF] Predicting. This can take a long time for a large prediction set. # 00:17:04 #
[ZEROCONF] Prediction done # 00:20:47 #
[ZEROCONF] Exporting the data # 00:20:47 #
[ZEROCONF] ##### Zeroconf Script Completed! ##### # 00:20:47 #

python evaluate-dataset-Adult.py 
[ZEROCONF]  # 00:27:22 #
[ZEROCONF] ######################################################################## # 00:27:22 #
[ZEROCONF] Accuracy score 84% # 00:27:22 #
[ZEROCONF] The below scores are calculated for predicting '1' category value # 00:27:22 #
[ZEROCONF] Precision: 65%, Recall: 73%, F1: 0.69 # 00:27:22 #
[ZEROCONF] Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall # 00:27:22 #
[ZEROCONF] [[10926  1509] # 00:27:22 #
[ZEROCONF]  [ 1020  2826]] # 00:27:22 #
[ZEROCONF] Baseline 3846 positives from 16281 overall = 23.6% # 00:27:22 #
[ZEROCONF] ######################################################################## # 00:27:22 #
[ZEROCONF]  # 00:27:22 #
</pre>
## Workarounds
these are not related to the autosklearn-zeroconf or auto-sklearn but rather general issues depending on your python and OS installation
### xgboost issues
#### complains about ELF header
<pre>pip uninstall xgboost; pip install --no-cache-dir -v xgboost==0.4a30</pre>
#### can not find libraries
<pre>conda install libgcc # for xgboost</pre>
alternatively search for them with 
<pre>sudo find / -name libgomp.so.1
/usr/lib/x86_64-linux-gnu/libgomp.so.1</pre> 
and explicitly add them to the libraries path
<pre>export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6":"/usr/lib/x86_64-linux-gnu/libgomp.so.1"; python zeroconf.py Titanic.h5 2>/dev/null|grep ZEROCONF</pre>
Also see https://github.com/automl/auto-sklearn/issues/247
