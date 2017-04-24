# -*- coding: utf-8 -*-
"""
Copyright 2017 PayPal
Created on Sun Oct 02 17:13:59 2016
@author: ekobylkin

This is an example on how to prepare data for autosklearn-zeroconf.
It is using a well known Titanic dataset from Kaggle https://www.kaggle.com/c/titanic .
"""
import pandas as pd
# Dowlnoad these files from Kaggle dataset
#https://www.kaggle.com/c/titanic/download/train.csv
#https://www.kaggle.com/c/titanic/download/test.csv
train = pd.read_csv(filepath_or_buffer='train.csv',sep=',', error_bad_lines=False, index_col=False)
test = pd.read_csv(filepath_or_buffer='test.csv',sep=',', error_bad_lines=False, index_col=False)

# We will use the test.csv data to make a prediction. You can compare the predicted values with the ground truth yourself.
test['Survived']=None # The empty target column tells autosklearn-zeroconf to use these cases for the prediction

dataframe=train.append(test)

# autosklearn-zeroconf requires cust_id and category (target or "y" variable) columns, the rest is optional
dataframe.rename(columns = {'PassengerId':'cust_id','Survived':'category'},inplace=True)

store = pd.HDFStore('Titanic.h5') # this is the file cache for the data
store['data'] = dataframe
store.close()
#Now run 'python zeroconf.py Titanic.h5' (python >=3.5)
