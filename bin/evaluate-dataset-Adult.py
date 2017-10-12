# -*- coding: utf-8 -*-
"""
Copyright 2017 Egor Kobylkin 
Created on Sun Apr 23 11:52:59 2017
@author: ekobylkin
This is an example on how to prepare data for autosklearn-zeroconf.
It is using a well known Adult (Salary) dataset from UCI https://archive.ics.uci.edu/ml/datasets/Adult .
"""
import pandas as pd

test = pd.read_csv(filepath_or_buffer='./data/adult.test.withid',sep=',', error_bad_lines=False, index_col=False)
#print(test)

prediction = pd.read_csv(filepath_or_buffer='./data/zeroconf-result.csv',sep=',', error_bad_lines=False, index_col=False)
#print(prediction)

df=pd.merge(test, prediction, how='inner', on=['cust_id',])

y_test=df['category']
y_hat=df['prediction']

from sklearn.metrics import (confusion_matrix, precision_score
, recall_score, f1_score, accuracy_score)
from time import time,sleep,strftime
def p(text):
    for line in str(text).splitlines():
        print ('[ZEROCONF] '+line+" # "+strftime("%H:%M:%S")+" #")

p("\n")
p("#"*72)
p("Accuracy score {0:2.0%}".format(accuracy_score(y_test, y_hat)))
p("The below scores are calculated for predicting '1' category value")
p("Precision: {0:2.0%}, Recall: {1:2.0%}, F1: {2:.2f}".format(
precision_score(y_test, y_hat),recall_score(y_test, y_hat),f1_score(y_test, y_hat)))
p("Confusion Matrix: https://en.wikipedia.org/wiki/Precision_and_recall")
p(confusion_matrix(y_test, y_hat))   
baseline_1 = str(sum(a for a in y_test))
baseline_all = str(len(y_test))
baseline_prcnt = "{0:2.0%}".format( float(sum(a for a in y_test)/len(y_test))) 
p("Baseline %s positives from %s overall = %1.1f%%" %
(sum(a for a in y_test), len(y_test), 100*sum(a for a in y_test)/len(y_test)))
p("#"*72)
p("\n")
