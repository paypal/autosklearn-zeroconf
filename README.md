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
To run autosklearn-zeroconf start <pre>python zeroconf.py your_dataframe.h5 2>/dev/null|grep ZEROCONF</pre> from command line.
The script was tested on Ubuntu and RedHat. It won't work on any WindowsOS because auto-sklearn doesn't support Windows.

## Data Format
The code uses a pandas dataframe format to manage the data. It is stored in the HDF5 file for convenience.

## Example
As an example you can run autosklearn-zeroconf on a "Census Income" dataset https://archive.ics.uci.edu/ml/datasets/Adult.
<pre>
python zeroconf-load-dataset-Adult.py
python zeroconf.py Adult.h5 2>/dev/null|grep ZEROCONF
</pre>
And then to evaluate the prediction stored in zerconf-result.csv against the test dataset file adult.test.withid 
<pre>python evaluate-dataset-Adult.py</pre>

## Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.
Alternatively you could use git clone
<pre>
sudo apt-get update && sudo apt-get install git && git clone https://github.com/paypal/autosklearn-zeroconf.git
</pre>

### Install auto-sklearn
<pre>
# A compiler (gcc) is needed to compile a few things the from auto-sklearn requirements.txt
# Chose just the line for your Linux flavor below
# On Ubuntu
sudo apt-get install gcc build-essential swig
# On RedHat
yum -y groupinstall 'Development Tools'
# if you want to use virtual environments
pip install virtualenv
virtualenv zeroconf -p /usr/bin/python3.5
source zeroconf/bin/activate
# requirements for auto-sklearn
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
# requrements for zeroconf
curl https://raw.githubusercontent.com/paypal/autosklearn-zeroconf/master/requirements.txt | xargs -n 1 -L 1 pip install
</pre>

<pre>
# If you have no Python environment installed, install Anaconda.
wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -O Anaconda3-Linux-x86_64.sh
chmod u+x Anaconda3-Linux-x86_64.sh
./Anaconda3-Linux-x86_64.sh
conda install libgcc

</pre>

## License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

## Example of the output
<pre>
python zeroconf.py Adult.h5 2>/dev/null|grep ZEROCONF
[ZEROCONF] Read dataset from the store # 00:31:12 #
[ZEROCONF] Values of y [  0.   1.  nan] # 00:31:12 #
[ZEROCONF] We need to protect NAs in y from the prediction dataset so we convert them to -1 # 00:31:12 #
[ZEROCONF] New values of y [ 0.  1. -1.] # 00:31:12 #
[ZEROCONF] Filling missing values in X with the most frequent values # 00:31:12 #
[ZEROCONF] Factorizing the X # 00:31:12 #
[ZEROCONF] Dataframe split into X and y # 00:31:12 #
[ZEROCONF] Preparing a sample to measure approx classifier run time and select features # 00:31:12 #
[ZEROCONF] Reserved 33% of the training dataset for validation (upto 33k rows) # 00:31:12 #
[ZEROCONF] Constructing preprocessor pipeline and transforming sample data # 00:31:12 #
[ZEROCONF] Running estimators on the sample # 00:31:14 #
[ZEROCONF] bernoulli_nb starting # 00:31:14 #
[ZEROCONF] bernoulli_nb training time: 0.03244924545288086 # 00:31:14 #
[ZEROCONF] decision_tree starting # 00:31:14 #
[ZEROCONF] decision_tree training time: 0.02585911750793457 # 00:31:14 #
[ZEROCONF] gaussian_nb starting # 00:31:14 #
[ZEROCONF] gaussian_nb training time: 0.03680753707885742 # 00:31:14 #
[ZEROCONF] multinomial_nb starting # 00:31:14 #
[ZEROCONF] multinomial_nb training time: 0.06878280639648438 # 00:31:15 #
[ZEROCONF] lda starting # 00:31:14 #
[ZEROCONF] lda training time: 0.09616875648498535 # 00:31:14 #
[ZEROCONF] liblinear_svc starting # 00:31:14 #
[ZEROCONF] liblinear_svc training time: 0.17355084419250488 # 00:31:15 #
[ZEROCONF] passive_aggressive starting # 00:31:14 #
[ZEROCONF] passive_aggressive training time: 0.28772568702697754 # 00:31:15 #
[ZEROCONF] sgd starting # 00:31:14 #
[ZEROCONF] sgd training time: 0.421036958694458 # 00:31:15 #
[ZEROCONF] k_nearest_neighbors starting # 00:31:14 #
[ZEROCONF] k_nearest_neighbors training time: 0.9100713729858398 # 00:31:15 #
[ZEROCONF] adaboost starting # 00:31:14 #
[ZEROCONF] adaboost training time: 1.1881482601165771 # 00:31:15 #
[ZEROCONF] xgradient_boosting starting # 00:31:15 #
[ZEROCONF] xgradient_boosting training time: 1.625457525253296 # 00:31:16 #
[ZEROCONF] extra_trees starting # 00:31:14 #
[ZEROCONF] extra_trees training time: 2.7511496543884277 # 00:31:17 #
[ZEROCONF] random_forest starting # 00:31:14 #
[ZEROCONF] random_forest training time: 2.8738672733306885 # 00:31:17 #
[ZEROCONF] gradient_boosting starting # 00:31:14 #
[ZEROCONF] gradient_boosting training time: 4.22829008102417 # 00:31:19 #
[ZEROCONF] Test classifier fit completed # 00:31:19 #
[ZEROCONF] per_run_time_limit=4 # 00:31:19 #
[ZEROCONF] Process pool size=2 # 00:31:19 #
[ZEROCONF] Starting autosklearn classifiers fiting on a 67% sample up to 67k rows # 00:31:19 #
[ZEROCONF] Max time allowance for a model 1 minute(s) # 00:31:19 #
[ZEROCONF] Overal run time is about 8 minute(s) # 00:31:19 #
[ZEROCONF] Starting seed=2 # 00:31:21 #
[ZEROCONF] Starting seed=3 # 00:31:22 #
[ZEROCONF] ####### Finished seed=2 # 00:34:35 #
[ZEROCONF] ####### Finished seed=3 # 00:34:38 #
[ZEROCONF] Multicore fit completed # 00:34:38 #
[ZEROCONF] Building ensemble # 00:34:38 #
[ZEROCONF] Ensemble built # 00:35:00 #
[ZEROCONF] Show models # 00:35:00 #
[ZEROCONF] [(0.300000, SimpleClassificationPipeline({'classifier:random_forest:n_estimators': 100, 'preprocessor:select_percentile_classification:percentile': 50.0, 'preprocessor:__choice__': 'select_percentile_classification', 'classifier:random_forest:criterion': 'entropy', 'classifier:random_forest:min_samples_leaf': 6, 'preprocessor:select_percentile_classification:score_func': 'chi2', 'classifier:random_forest:max_features': 4.839202954184717, 'balancing:strategy': 'weighting', 'imputation:strategy': 'mean', 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_samples_split': 6, 'classifier:random_forest:bootstrap': 'True'}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] (0.200000, SimpleClassificationPipeline({'classifier:xgradient_boosting:scale_pos_weight': 1, 'classifier:xgradient_boosting:n_estimators': 342, 'one_hot_encoding:minimum_fraction': 0.2547323861133189, 'classifier:xgradient_boosting:max_depth': 3, 'preprocessor:__choice__': 'no_preprocessing', 'balancing:strategy': 'none', 'rescaling:__choice__': 'minmax', 'imputation:strategy': 'median', 'classifier:xgradient_boosting:learning_rate': 0.04534535307037642, 'classifier:xgradient_boosting:reg_lambda': 1, 'classifier:xgradient_boosting:colsample_bylevel': 1, 'classifier:xgradient_boosting:max_delta_step': 0, 'classifier:xgradient_boosting:subsample': 0.6025298628426271, 'classifier:__choice__': 'xgradient_boosting', 'classifier:xgradient_boosting:reg_alpha': 0, 'classifier:xgradient_boosting:gamma': 0, 'classifier:xgradient_boosting:colsample_bytree': 1, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:xgradient_boosting:base_score': 0.5, 'classifier:xgradient_boosting:min_child_weight': 10}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] (0.200000, SimpleClassificationPipeline({'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:min_samples_leaf': 6, 'classifier:random_forest:max_features': 1.9992029381707617, 'balancing:strategy': 'weighting', 'imputation:strategy': 'mean', 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_samples_split': 8, 'classifier:random_forest:bootstrap': 'True'}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:n_estimators': 100, 'one_hot_encoding:minimum_fraction': 0.010000000000000004, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:min_samples_leaf': 1, 'classifier:random_forest:max_features': 1.0, 'balancing:strategy': 'none', 'imputation:strategy': 'mean', 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_samples_split': 8, 'classifier:random_forest:bootstrap': 'True'}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:min_samples_leaf': 6, 'classifier:random_forest:max_features': 3.718593691792686, 'balancing:strategy': 'weighting', 'imputation:strategy': 'mean', 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_samples_split': 8, 'classifier:random_forest:bootstrap': 'True'}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] (0.100000, SimpleClassificationPipeline({'classifier:random_forest:n_estimators': 100, 'preprocessor:__choice__': 'no_preprocessing', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:min_samples_leaf': 6, 'classifier:random_forest:max_features': 3.6470980321678206, 'balancing:strategy': 'weighting', 'imputation:strategy': 'mean', 'one_hot_encoding:use_minimum_fraction': 'False', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:max_depth': 'None', 'rescaling:__choice__': 'minmax', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_samples_split': 6, 'classifier:random_forest:bootstrap': 'True'}, # 00:35:00 #
[ZEROCONF] dataset_properties={ # 00:35:00 #
[ZEROCONF]   'task': 1, # 00:35:00 #
[ZEROCONF]   'multilabel': False, # 00:35:00 #
[ZEROCONF]   'target_type': 'classification', # 00:35:00 #
[ZEROCONF]   'signed': False, # 00:35:00 #
[ZEROCONF]   'multiclass': False, # 00:35:00 #
[ZEROCONF]   'sparse': False})), # 00:35:00 #
[ZEROCONF] ] # 00:35:00 #
[ZEROCONF] Validating # 00:35:00 #
[ZEROCONF] Predicting on validation set # 00:35:00 #
[ZEROCONF]  # 00:35:02 #
[ZEROCONF] ######################################################################## # 00:35:02 #
[ZEROCONF] Accuracy score 85% # 00:35:02 #
[ZEROCONF] The below scores are calculated for predicting '1' category value # 00:35:02 #
[ZEROCONF] Precision: 65%, Recall: 80%, F1: 0.71 # 00:35:02 #
[ZEROCONF] Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall # 00:35:02 #
[ZEROCONF] [[7026 1132] # 00:35:02 #
[ZEROCONF]  [ 528 2060]] # 00:35:02 #
[ZEROCONF] Baseline 2588 positives from 10746 overall = 24.1% # 00:35:02 #
[ZEROCONF] ######################################################################## # 00:35:02 #
[ZEROCONF]  # 00:35:02 #
[ZEROCONF] Dataframe split into X and y # 00:35:02 #
[ZEROCONF] Re-fitting the model ensemble on full known dataset to prepare for prediciton. This can take a long time. # 00:35:02 #
[ZEROCONF] Predicting. This can take a long time for a large prediction set. # 00:35:50 #
[ZEROCONF] Prediction done # 00:35:54 #
[ZEROCONF] Exporting the data # 00:35:54 #
[ZEROCONF] ##### Zeroconf Script Completed! ##### # 00:35:54 #
</pre>

<pre>
python evaluate-dataset-Adult.py 
[ZEROCONF]  # 00:37:43 #
[ZEROCONF] ######################################################################## # 00:37:43 #
[ZEROCONF] Accuracy score 85% # 00:37:43 #
[ZEROCONF] The below scores are calculated for predicting '1' category value # 00:37:43 #
[ZEROCONF] Precision: 65%, Recall: 78%, F1: 0.71 # 00:37:43 #
[ZEROCONF] Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall # 00:37:43 #
[ZEROCONF] [[10835  1600] # 00:37:43 #
[ZEROCONF]  [  860  2986]] # 00:37:43 #
[ZEROCONF] Baseline 3846 positives from 16281 overall = 23.6% # 00:37:43 #
[ZEROCONF] ######################################################################## # 00:37:43 #
[ZEROCONF]  # 00:37:43 #
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
