## What is autosklearn-zeroconf
The autosklearn-zeroconf file takes a dataframe of any size and trains [auto-sklearn](https://github.com/automl/auto-sklearn) binary classifier ensemble. No configuration is needed as the name suggests. Auto-sklearn is the recent [AutoML Challenge](http://www.kdnuggets.com/2016/08/winning-automl-challenge-auto-sklearn.html) winner [more @microsoft.com](https://www.microsoft.com/en-us/research/blog/automl-challenge-leap-forward-machine-learning-competitions/).

As a result of using automl-zeroconf running auto-sklearn becomes a "fire and forget" type of operation. It greatly increases the utility and decreases turnaround time for experiments.

The main value proposition is that a data analyst or a data savvy business user can quickly run the iterations on the data (actual sources and feature design) side and on the ML side not a bit has to be changed. So it's a great tool for people not doing hardcore data science full time. Up to 90% of (marketing) data analysts may fall into this target group currently. 

## How Does It Work
To keep the training time reasonable autosklearn-zeroconf samples the data and tests all the models from autosklearn library on it once. The results of the test (duration) is used to calculate the per_run_time_limit, time_left_for_this_task and number of seeds parameters for autosklearn. The code also converts the pandas dataframe into a form that autosklearn can handle (categorical and float datatypes).
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
To run autosklearn-zeroconf start <pre>python bin/zeroconf.py -d your_dataframe.h5</pre> from command line.
The script was tested on Ubuntu and RedHat. It won't work on any WindowsOS because auto-sklearn doesn't support Windows.

## Data Format
The code uses a pandas dataframe format to manage the data. It is stored in the HDF5 .h5 file for convenience. (Python module "tables")

## Example
As an example you can run autosklearn-zeroconf on a "Census Income" dataset https://archive.ics.uci.edu/ml/datasets/Adult.
<pre>python ./bin/zeroconf.py -d ./data/Adult.h5</pre>
And then to evaluate the prediction stored in zerconf-result.csv against the test dataset file adult.test.withid 
<pre>python ./bin/evaluate-dataset-Adult.py</pre>

## Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.
Alternatively you could use git clone
<pre>
sudo apt-get update && sudo apt-get install git && git clone https://github.com/paypal/autosklearn-zeroconf.git
</pre>

### Happy path installation on Ubuntu 18.04LTS
<pre>
sudo apt-get update && sudo apt-get install git gcc build-essential swig python-pip virtualenv python3-dev
git clone https://github.com/paypal/autosklearn-zeroconf.git
pip install virtualenv
virtualenv zeroconf -p /usr/bin/python3.6
source zeroconf/bin/activate
curl https://raw.githubusercontent.com/paypal/autosklearn-zeroconf/master/requirements.txt | xargs -n 1 -L 1 pip install
git clone https://github.com/paypal/autosklearn-zeroconf.git
cd autosklearn-zeroconf/ && python ./bin/zeroconf.py -d ./data/Adult.h5 2>/dev/null
</pre>

## License
autosklearn-zeroconf is licensed under the [BSD 3-Clause License (Revised)](LICENSE.txt)

## Example of the output
<pre>
python zeroconf.py -d ./data/Adult.h5 2>/dev/null | grep [ZEROCONF]

2017-10-11 10:52:15,893 - [ZEROCONF] - zeroconf.py - INFO - Program Call Parameter (Arguments and Parameter File Values):
2017-10-11 10:52:15,893 - [ZEROCONF] - zeroconf.py - INFO -    basedir: /home/ulrich/PycharmProjects/autosklearn-zeroconf
2017-10-11 10:52:15,893 - [ZEROCONF] - zeroconf.py - INFO -    data_file: /home/ulrich/PycharmProjects/autosklearn-zeroconf/data/Adult.h5
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    id_field: cust_id
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    max_classifier_time_budget: 1200
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    max_sample_size: 100000
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    memory_limit: 15000
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    parameter_file: /home/ulrich/PycharmProjects/autosklearn-zeroconf/parameter/default.yml
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    proc: zeroconf.py
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    resultfile: /home/ulrich/PycharmProjects/autosklearn-zeroconf/data/zeroconf-result.csv
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    runid: 20171011105215
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    runtype: Fresh Run Start
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    target_field: category
2017-10-11 10:52:15,894 - [ZEROCONF] - zeroconf.py - INFO -    workdir: /home/ulrich/PycharmProjects/autosklearn-zeroconf/work/20171011105215
2017-10-11 10:52:15,944 - [ZEROCONF] - zeroconf.py - INFO - Read dataset from the store
2017-10-11 10:52:15,945 - [ZEROCONF] - zeroconf.py - INFO - Values of y [  0.   1.  nan]
2017-10-11 10:52:15,945 - [ZEROCONF] - zeroconf.py - INFO - We need to protect NAs in y from the prediction dataset so we convert them to -1
2017-10-11 10:52:15,946 - [ZEROCONF] - zeroconf.py - INFO - New values of y [ 0.  1. -1.]
2017-10-11 10:52:15,946 - [ZEROCONF] - zeroconf.py - INFO - Filling missing values in X with the most frequent values
2017-10-11 10:52:16,043 - [ZEROCONF] - zeroconf.py - INFO - Factorizing the X
2017-10-11 10:52:16,176 - [ZEROCONF] - x_y_dataframe_split - INFO - Dataframe split into X and y
2017-10-11 10:52:16,178 - [ZEROCONF] - zeroconf.py - INFO - Preparing a sample to measure approx classifier run time and select features
2017-10-11 10:52:16,191 - [ZEROCONF] - zeroconf.py - INFO - train size:21815
2017-10-11 10:52:16,191 - [ZEROCONF] - zeroconf.py - INFO - test size:10746
2017-10-11 10:52:16,192 - [ZEROCONF] - zeroconf.py - INFO - Reserved 33% of the training dataset for validation (upto 33k rows)
2017-10-11 10:52:16,209 - [ZEROCONF] - max_estimators_fit_duration - INFO - Constructing preprocessor pipeline and transforming sample data
2017-10-11 10:52:18,712 - [ZEROCONF] - max_estimators_fit_duration - INFO - Running estimators on the sample
2017-10-11 10:52:18,729 - [ZEROCONF] - zeroconf.py - INFO - adaboost starting
2017-10-11 10:52:18,734 - [ZEROCONF] - zeroconf.py - INFO - bernoulli_nb starting
2017-10-11 10:52:18,761 - [ZEROCONF] - zeroconf.py - INFO - extra_trees starting
2017-10-11 10:52:18,769 - [ZEROCONF] - zeroconf.py - INFO - decision_tree starting
2017-10-11 10:52:18,780 - [ZEROCONF] - zeroconf.py - INFO - gaussian_nb starting
2017-10-11 10:52:18,800 - [ZEROCONF] - zeroconf.py - INFO - bernoulli_nb training time: 0.06455278396606445
2017-10-11 10:52:18,802 - [ZEROCONF] - zeroconf.py - INFO - gradient_boosting starting
2017-10-11 10:52:18,808 - [ZEROCONF] - zeroconf.py - INFO - k_nearest_neighbors starting
2017-10-11 10:52:18,809 - [ZEROCONF] - zeroconf.py - INFO - decision_tree training time: 0.03273773193359375
2017-10-11 10:52:18,826 - [ZEROCONF] - zeroconf.py - INFO - lda starting
2017-10-11 10:52:18,845 - [ZEROCONF] - zeroconf.py - INFO - liblinear_svc starting
2017-10-11 10:52:18,867 - [ZEROCONF] - zeroconf.py - INFO - gaussian_nb training time: 0.08569979667663574
2017-10-11 10:52:18,882 - [ZEROCONF] - zeroconf.py - INFO - multinomial_nb starting
2017-10-11 10:52:18,905 - [ZEROCONF] - zeroconf.py - INFO - passive_aggressive starting
2017-10-11 10:52:18,943 - [ZEROCONF] - zeroconf.py - INFO - random_forest starting
2017-10-11 10:52:18,971 - [ZEROCONF] - zeroconf.py - INFO - sgd starting
2017-10-11 10:52:19,012 - [ZEROCONF] - zeroconf.py - INFO - lda training time: 0.17656564712524414
2017-10-11 10:52:19,023 - [ZEROCONF] - zeroconf.py - INFO - multinomial_nb training time: 0.13777780532836914
2017-10-11 10:52:19,124 - [ZEROCONF] - zeroconf.py - INFO - liblinear_svc training time: 0.27405595779418945
2017-10-11 10:52:19,416 - [ZEROCONF] - zeroconf.py - INFO - passive_aggressive training time: 0.508676290512085
2017-10-11 10:52:19,473 - [ZEROCONF] - zeroconf.py - INFO - sgd training time: 0.49777913093566895
2017-10-11 10:52:20,471 - [ZEROCONF] - zeroconf.py - INFO - adaboost training time: 1.7392246723175049
2017-10-11 10:52:20,625 - [ZEROCONF] - zeroconf.py - INFO - k_nearest_neighbors training time: 1.8141863346099854
2017-10-11 10:52:22,258 - [ZEROCONF] - zeroconf.py - INFO - extra_trees training time: 3.4934401512145996
2017-10-11 10:52:22,696 - [ZEROCONF] - zeroconf.py - INFO - random_forest training time: 3.7496204376220703
2017-10-11 10:52:24,215 - [ZEROCONF] - zeroconf.py - INFO - gradient_boosting training time: 5.41023063659668
2017-10-11 10:52:24,230 - [ZEROCONF] - max_estimators_fit_duration - INFO - Test classifier fit completed
2017-10-11 10:52:24,239 - [ZEROCONF] - zeroconf.py - INFO - per_run_time_limit=5
2017-10-11 10:52:24,239 - [ZEROCONF] - zeroconf.py - INFO - Process pool size=2
2017-10-11 10:52:24,240 - [ZEROCONF] - zeroconf.py - INFO - Starting autosklearn classifiers fiting on a 67% sample up to 67k rows
2017-10-11 10:52:24,252 - [ZEROCONF] - train_multicore - INFO - Max time allowance for a model 1 minute(s)
2017-10-11 10:52:24,252 - [ZEROCONF] - train_multicore - INFO - Overal run time is about 10 minute(s)
2017-10-11 10:52:24,255 - [ZEROCONF] - train_multicore - INFO - Multicore process 2 started
2017-10-11 10:52:24,258 - [ZEROCONF] - train_multicore - INFO - Multicore process 3 started
2017-10-11 10:52:24,276 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Start AutoSklearnClassifier seed=2
2017-10-11 10:52:24,278 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Start AutoSklearnClassifier seed=3
2017-10-11 10:52:24,295 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Done AutoSklearnClassifier seed=3
2017-10-11 10:52:24,297 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Done AutoSklearnClassifier seed=2
2017-10-11 10:52:26,299 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Starting seed=2
2017-10-11 10:52:27,298 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - Starting seed=3
2017-10-11 10:56:30,949 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - ####### Finished seed=2
2017-10-11 10:56:31,600 - [ZEROCONF] - spawn_autosklearn_classifier - INFO - ####### Finished seed=3
2017-10-11 10:56:31,614 - [ZEROCONF] - train_multicore - INFO - Multicore fit completed
2017-10-11 10:56:31,626 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - Building ensemble
2017-10-11 10:56:31,626 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - Done AutoSklearnClassifier - seed:1
2017-10-11 10:56:54,017 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - Ensemble built - seed:1
2017-10-11 10:56:54,017 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - Show models - seed:1
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - [(0.400000, SimpleClassificationPipeline({'classifier:__choice__': 'adaboost', 'one_hot_encoding:use_minimum_fraction': 'True', 'preprocessor:select_percentile_classification:percentile': 85.5410729966473, 'classifier:adaboost:n_estimators': 88, 'one_hot_encoding:minimum_fraction': 0.01805038589303469, 'rescaling:__choice__': 'minmax', 'balancing:strategy': 'weighting', 'preprocessor:__choice__': 'select_percentile_classification', 'classifier:adaboost:max_depth': 1, 'classifier:adaboost:learning_rate': 0.10898092508755285, 'preprocessor:select_percentile_classification:score_func': 'chi2', 'imputation:strategy': 'most_frequent', 'classifier:adaboost:algorithm': 'SAMME.R'},
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - dataset_properties={
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'task': 1,
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'signed': False,
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'sparse': False,
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multiclass': False,
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'target_type': 'classification',
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multilabel': False})),
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - (0.300000, SimpleClassificationPipeline({'classifier:__choice__': 'random_forest', 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:min_samples_leaf': 4, 'classifier:random_forest:max_depth': 'None', 'classifier:random_forest:min_samples_split': 16, 'classifier:random_forest:bootstrap': 'False', 'one_hot_encoding:minimum_fraction': 0.1453954841364777, 'rescaling:__choice__': 'none', 'balancing:strategy': 'none', 'preprocessor:__choice__': 'select_percentile_classification', 'preprocessor:select_percentile_classification:percentile': 96.35414862145892, 'preprocessor:select_percentile_classification:score_func': 'chi2', 'imputation:strategy': 'mean', 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:max_features': 3.342759426984195, 'classifier:random_forest:n_estimators': 100},
2017-10-11 10:56:54,596 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - dataset_properties={
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'task': 1,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'signed': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'sparse': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multiclass': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'target_type': 'classification',
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multilabel': False})),
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - (0.200000, SimpleClassificationPipeline({'classifier:extra_trees:min_weight_fraction_leaf': 0.0, 'classifier:__choice__': 'extra_trees', 'classifier:extra_trees:n_estimators': 100, 'classifier:extra_trees:bootstrap': 'True', 'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 5, 'classifier:extra_trees:min_samples_leaf': 10, 'rescaling:__choice__': 'minmax', 'classifier:extra_trees:max_depth': 'None', 'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'True', 'preprocessor:extra_trees_preproc_for_classification:criterion': 'gini', 'classifier:extra_trees:max_features': 4.413198608615693, 'classifier:extra_trees:criterion': 'gini', 'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100, 'classifier:extra_trees:min_samples_split': 16, 'one_hot_encoding:use_minimum_fraction': 'False', 'balancing:strategy': 'weighting', 'preprocessor:__choice__': 'extra_trees_preproc_for_classification', 'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 1, 'preprocessor:extra_trees_preproc_for_classification:max_features': 1.4824479003506632, 'imputation:strategy': 'median', 'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0, 'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None'},
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - dataset_properties={
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'task': 1,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'signed': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'sparse': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multiclass': False,
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'target_type': 'classification',
2017-10-11 10:56:54,597 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multilabel': False})),
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - (0.100000, SimpleClassificationPipeline({'classifier:extra_trees:min_weight_fraction_leaf': 0.0, 'classifier:__choice__': 'extra_trees', 'classifier:extra_trees:n_estimators': 100, 'classifier:extra_trees:bootstrap': 'True', 'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 16, 'classifier:extra_trees:min_samples_leaf': 10, 'rescaling:__choice__': 'minmax', 'classifier:extra_trees:max_depth': 'None', 'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'True', 'preprocessor:extra_trees_preproc_for_classification:criterion': 'gini', 'classifier:extra_trees:max_features': 4.16852017424403, 'classifier:extra_trees:criterion': 'gini', 'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100, 'classifier:extra_trees:min_samples_split': 16, 'one_hot_encoding:use_minimum_fraction': 'False', 'balancing:strategy': 'weighting', 'preprocessor:__choice__': 'extra_trees_preproc_for_classification', 'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 1, 'preprocessor:extra_trees_preproc_for_classification:max_features': 1.5781770540350555, 'imputation:strategy': 'median', 'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0, 'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None'},
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - dataset_properties={
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'task': 1,
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'signed': False,
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'sparse': False,
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multiclass': False,
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'target_type': 'classification',
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO -   'multilabel': False})),
2017-10-11 10:56:54,598 - [ZEROCONF] - zeroconf_fit_ensemble - INFO - ]
2017-10-11 10:56:54,613 - [ZEROCONF] - zeroconf.py - INFO - Validating
2017-10-11 10:56:54,613 - [ZEROCONF] - zeroconf.py - INFO - Predicting on validation set
2017-10-11 10:56:57,373 - [ZEROCONF] - zeroconf.py - INFO - ########################################################################
2017-10-11 10:56:57,374 - [ZEROCONF] - zeroconf.py - INFO - Accuracy score 84%
2017-10-11 10:56:57,374 - [ZEROCONF] - zeroconf.py - INFO - The below scores are calculated for predicting '1' category value
2017-10-11 10:56:57,379 - [ZEROCONF] - zeroconf.py - INFO - Precision: 64%, Recall: 77%, F1: 0.70
2017-10-11 10:56:57,379 - [ZEROCONF] - zeroconf.py - INFO - Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall
2017-10-11 10:56:57,386 - [ZEROCONF] - zeroconf.py - INFO - [7058 1100]
2017-10-11 10:56:57,386 - [ZEROCONF] - zeroconf.py - INFO - [ 603 1985]
2017-10-11 10:56:57,392 - [ZEROCONF] - zeroconf.py - INFO - Baseline 2588 positives from 10746 overall = 24.1%
2017-10-11 10:56:57,392 - [ZEROCONF] - zeroconf.py - INFO - ########################################################################
2017-10-11 10:56:57,404 - [ZEROCONF] - x_y_dataframe_split - INFO - Dataframe split into X and y
2017-10-11 10:56:57,405 - [ZEROCONF] - zeroconf.py - INFO - Re-fitting the model ensemble on full known dataset to prepare for prediciton. This can take a long time.
2017-10-11 10:58:39,836 - [ZEROCONF] - zeroconf.py - INFO - Predicting. This can take a long time for a large prediction set.
2017-10-11 10:58:45,221 - [ZEROCONF] - zeroconf.py - INFO - Prediction done
2017-10-11 10:58:45,223 - [ZEROCONF] - zeroconf.py - INFO - Exporting the data
2017-10-11 10:58:45,267 - [ZEROCONF] - zeroconf.py - INFO - ##### Zeroconf Script Completed! #####
2017-10-11 10:58:45,268 - [ZEROCONF] - zeroconf.py - INFO - Clean up / Delete work directory: /home/ulrich/PycharmProjects/autosklearn-zeroconf/work/20171011105215

Process finished with exit code 0
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

### Install auto-sklearn
<pre>
# A compiler (gcc) is needed to compile a few things the from auto-sklearn requirements.txt
# Chose just the line for your Linux flavor below

# On Ubuntu
sudo apt-get install gcc build-essential swig

# On CentOS 7-1611 http://www.osboxes.org/centos/ https://drive.google.com/file/d/0B_HAFnYs6Ur-bl8wUWZfcHVpMm8/view?usp=sharing
sudo yum -y update 
sudo reboot
sudo yum install epel-release python34 python34-devel python34-setuptools
sudo yum -y groupinstall 'Development Tools'

# auto-sklearn requires swig 3.0 
wget downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz -O swig-3.0.12.tar.gz
tar xf swig-3.0.12.tar.gz 
cd swig-3.0.12 
./configure --without-pcre
make
sudo make install
cd ..

sudo easy_install-3.4 pip
# if you want to use virtual environments
sudo pip3 install virtualenv
virtualenv zeroconf -p /usr/bin/python3.4
source zeroconf/bin/activate

curl https://raw.githubusercontent.com/paypal/autosklearn-zeroconf/master/requirements.txt | xargs -n 1 -L 1 pip install
</pre>

# Contributors
Egor Kobylkin, Ulrich Arndt
