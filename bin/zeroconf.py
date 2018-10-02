# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""
Copyright 2017 PayPal
Created on Mon Feb 27 19:11:59 PST 2017
@author: ekobylkin
@version 0.2
@author: ulrich arndt - data2knowledge
@update: 2017-09-27
"""

import argparse
import numpy as np
import os
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)

import utility as utl
import dataTransformationProcessing as dt

parameter = utl.init_process(__file__)

###########################################################
# define the command line argument parser
###########################################################
# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser(
    description='zero configuration predictic modeling script. Requires a pandas HDFS dataframe file ' + \
                'and a yaml parameter file as input as input')
parser.add_argument('-d',
                    '--data_file',
                    nargs=1,
                    help='input pandas HDFS dataframe .h5 with an unique indentifier and a target column\n' +
                         'as well as additional data columns\n'
                         'default values are cust_id and category or need to be defined in an\n' +
                         'optional parameter file '
                    )
parser.add_argument('-p',
                    '--param_file',
                    help='input yaml parameter file'
                    )

args = parser.parse_args()
logger = utl.get_logger(os.path.basename(__file__))
logger.info("Program started with the following arguments:")
logger.info(args)

###########################################################
# set dir to project dir
###########################################################
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
os.chdir(dname)

###########################################################
# file check for the parameter
###########################################################
param_file = ''
if args.param_file:
    param_file = args.param_file[0]
else:
    param_file = os.path.abspath("./parameter/default.yml")
    logger.info("Using the default parameter file: " + param_file)
if (not (os.path.isfile(param_file))):
    msg = 'the input parameter file: ' + param_file + ' does not exist!'
    logger.error(msg)
    exit(8)

data_file = ''
if args.data_file:
    data_file = args.data_file[0]
else:
    msg = "A data file is mandatory!"
    logger.error(msg)
    exit(8)
if (not (os.path.isfile(data_file))):
    msg = 'the input parameter file: ' + data_file + ' does not exist!'
    logger.error(msg)
    exit(8)

parameter = utl.read_parameter(param_file, parameter)

parameter["data_file"] = os.path.abspath(data_file)
parameter["basedir"] = os.path.abspath(parameter["basedir"])
parameter["parameter_file"] = os.path.abspath(param_file)
parameter["resultfile"] = os.path.abspath(parameter["resultfile"])


###########################################################
# set base dir
###########################################################
os.chdir(parameter["basedir"])
logger.info("Set basedir to: " + parameter["basedir"])

logger = utl.get_logger(os.path.basename(__file__))

logger.info("Program Call Parameter (Arguments and Parameter File Values):")
for key in sorted(parameter.keys()):
    logger.info("   " + key + ": " + str(parameter[key]))

work_dir = parameter["workdir"]
result_filename = parameter["resultfile"]
atsklrn_tempdir = os.path.join(work_dir, 'atsklrn_tmp')
shutil.rmtree(atsklrn_tempdir, ignore_errors=True)  # cleanup - remove temp directory


# if the memory limit is lower the model can fail and the whole process will crash
memory_limit = parameter["memory_limit"]  # MB
global max_classifier_time_budget
max_classifier_time_budget = parameter["max_classifier_time_budget"]  # but 10 minutes is usually more than enough
max_sample_size = parameter["max_sample_size"]  # so that the classifiers fit method completes in a reasonable time

dataframe = dt.read_dataframe_h5(data_file, logger)

logger.info("Values of y " + str(dataframe[parameter["target_field"]].unique()))
logger.info("We need to protect NAs in y from the prediction dataset so we convert them to -1")
dataframe[parameter["target_field"]] = dataframe[parameter["target_field"]].fillna(-1)
logger.info("New values of y " + str(dataframe[parameter["target_field"]].unique()))

logger.info("Filling missing values in X with the most frequent values")
dataframe = dataframe.fillna(dataframe.mode().iloc[0])

logger.info("Factorizing the X")
# we need this list of original dtypes for the Autosklearn fit, create it before categorisation or split
col_dtype_dict = {col: ('Numerical' if np.issubdtype(dataframe[col].dtype, np.number) else 'Categorical')
                  for col in dataframe.columns if col not in [parameter["id_field"], parameter["target_field"]]}

# http://stackoverflow.com/questions/25530504/encoding-column-labels-in-pandas-for-machine-learning
# http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn?rq=1
# https://github.com/automl/auto-sklearn/issues/121#issuecomment-251459036

for col in dataframe.select_dtypes(exclude=[np.number]).columns:
    if col not in [parameter["id_field"], parameter["target_field"]]:
        dataframe[col] = dataframe[col].astype('category').cat.codes

df_unknown = dataframe[dataframe[parameter["target_field"]] == -1]  # 'None' gets categorzized into -1
df_known = dataframe[dataframe[parameter["target_field"]] != -1]  # not [0,1] for multiclass labeling compartibility
logger.debug("Length of unknown dataframe:" + str(len(df_unknown)))
logger.debug("Length of known dataframe:" + str(len(df_known)))

del dataframe

X, y = dt.x_y_dataframe_split(df_known, parameter)

logger.info("Preparing a sample to measure approx classifier run time and select features")
dataset_size = df_known.shape[0]

if dataset_size > max_sample_size:
    sample_factor = dataset_size / float(max_sample_size)
    logger.info("Sample factor =" + str(sample_factor))
    X_sample, y_sample = dt.x_y_dataframe_split(df_known.sample(max_sample_size, random_state=42), parameter)
    X_train, X_test, y_train, y_test = dt.train_test_split(X.copy(), y, stratify=y, test_size=33000,
                                                           random_state=42)  # no need for larger test
else:
    sample_factor = 1
    X_sample, y_sample = X.copy(), y
    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y, stratify=y, test_size=0.33, random_state=42)
logger.info("train size:" + str(len(X_train)))
logger.info("test size:" + str(len(X_test)))
logger.info("Reserved 33% of the training dataset for validation (upto 33k rows)")

per_run_time_limit = dt.max_estimators_fit_duration(X_train.values, y_train, max_classifier_time_budget, logger,
                                                    sample_factor)
logger.info("per_run_time_limit=" + str(per_run_time_limit))
pool_size = dt.define_pool_size(int(memory_limit))
logger.info("Process pool size=" + str(pool_size))
feat_type = [col_dtype_dict[col] for col in X.columns]
logger.info("Starting autosklearn classifiers fiting on a 67% sample up to 67k rows")
dt.train_multicore(X_train.values, y_train, feat_type, int(memory_limit), atsklrn_tempdir, pool_size,
                   per_run_time_limit)

ensemble = dt.zeroconf_fit_ensemble(y_train, atsklrn_tempdir)

logger = utl.get_logger(os.path.basename(__file__))
logger.info("Validating")
logger.info("Predicting on validation set")
y_hat = ensemble.predict(X_test.values)

logger.info("#" * 72)
logger.info("Accuracy score {0:2.0%}".format(accuracy_score(y_test, y_hat)))
logger.info("The below scores are calculated for predicting '1' category value")
logger.info("Precision: {0:2.0%}, Recall: {1:2.0%}, F1: {2:.2f}".format(
    precision_score(y_test, y_hat), recall_score(y_test, y_hat), f1_score(y_test, y_hat)))
#############################
## Print COnfusion Matrix
#############################
logger.info("Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall")
cm = confusion_matrix(y_test, y_hat)
for row in cm:
    logger.info(row)

baseline_1 = str(sum(a for a in y_test))
baseline_all = str(len(y_test))
baseline_prcnt = "{0:2.0%}".format(float(sum(a for a in y_test) / len(y_test)))
logger.info("Baseline %s positives from %s overall = %1.1f%%" %
            (sum(a for a in y_test), len(y_test), 100 * sum(a for a in y_test) / len(y_test)))
logger.info("#" * 72)

if df_unknown.shape[0] == 0:  # if there is nothing to predict we can stop already
    logger.info("##### Nothing to predict. Prediction dataset is empty. #####")
    exit(0)

X_unknown, y_unknown, row_id_unknown = dt.x_y_dataframe_split(df_unknown, parameter, id=True)

logger.info("Re-fitting the model ensemble on full known dataset to prepare for prediciton. This can take a long time.")
try:
    ensemble.refit(X.copy().values, y)
except Exception as e:
    logger.info("Refit failed, reshuffling the rows, restarting")
    logger.info(e)
    try:
        X2 = X.copy().values
        indices = np.arange(X2.shape[0])
        np.random.shuffle(indices)  # a workaround to algoritm shortcomings
        X2 = X2[indices]
        y = y[indices]
        ensemble.refit(X2, y)
    except Exception as e:
        logger.info("Second refit failed")
        logger.info(e)
        logger.info(
            " WORKAROUND: because Refitting fails due to an upstream bug https://github.com/automl/auto-sklearn/issues/263")
        logger.info(" WORKAROUND: we are fitting autosklearn classifiers a second time, now on the full dataset")
        dt.train_multicore(X.values, y, feat_type, int(memory_limit), atsklrn_tempdir, pool_size, per_run_time_limit)
        ensemble = dt.zeroconf_fit_ensemble(y_train, atsklrn_tempdir)

logger.info("Predicting. This can take a long time for a large prediction set.")
try:
    y_pred = ensemble.predict(X_unknown.copy().values)
    logger.info("Prediction done")
except Exception as e:
    logger.info(e)
    logger.info(
        " WORKAROUND: because REfitting fails due to an upstream bug https://github.com/automl/auto-sklearn/issues/263")
    logger.info(" WORKAROUND: we are fitting autosklearn classifiers a second time, now on the full dataset")
    dt.train_multicore(X.values, y, feat_type, int(memory_limit), atsklrn_tempdir, pool_size, per_run_time_limit)
    ensemble = dt.zeroconf_fit_ensemble(y_train, atsklrn_tempdir)
    logger.info("Predicting. This can take a long time for a large prediction set.")
    try:
        y_pred = ensemble.predict(X_unknown.copy().values)
        logger.info("Prediction done")
    except Exception as e:
        logger.info("##### Prediction failed, exiting! #####")
        logger.info(e)
        exit(2)

result_df = pd.DataFrame(
    {parameter["id_field"]: row_id_unknown, 'prediction': pd.Series(y_pred, index=row_id_unknown.index)})
logger.info("Exporting the data")
result_df.to_csv(result_filename, index=False, header=True)
logger.info("##### Zeroconf Script Completed! #####")
utl.end_proc_success(parameter, logger)
