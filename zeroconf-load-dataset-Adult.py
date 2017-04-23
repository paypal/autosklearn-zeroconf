# -*- coding: utf-8 -*-
"""
Copyright 2017 Egor Kobylkin 
Created on Sun Apr 23 11:52:59 2017
@author: ekobylkin
This is an example on how to prepare data for autosklearn-zeroconf.
It is using a well known Adult (Salary) dataset from UCI https://archive.ics.uci.edu/ml/datasets/Adult .
"""
import pandas as pd
# Dowlnoad these files from Kaggle dataset
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
train = pd.read_csv(filepath_or_buffer='adult.data',sep=',', error_bad_lines=False, index_col=False)
columns=[
'age',
'workclass',
'fnlwgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country',
'category'
]
train.columns=columns
train['cust_id']=train.index
category_mapping={' >50K':1,' <=50K':0}
train['category']= train['category'].map(category_mapping) 
#test = pd.read_csv(filepath_or_buffer='adult.test',sep=',', error_bad_lines=False, index_col=False)
print(train)
#exit()
# We will use the test.csv data to make a prediction. You can compare the predicted values with the ground truth yourself.
#test['Survived']=None # The empty target column tells autosklearn-zeroconf to use these cases for the prediction

#dataframe=train.append(test)

# autosklearn-zeroconf requires cust_id and category (target or "y" variable) columns, the rest is optional
#dataframe.rename(columns = {'PassengerId':'cust_id','Survived':'category'},inplace=True)
dataframe=train
store = pd.HDFStore('Adult.h5') # this is the file cache for the data
store['data'] = dataframe
store.close()
#Now run 'python zeroconf.py Titanic.h5' (python >=3.5)
