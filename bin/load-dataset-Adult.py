# -*- coding: utf-8 -*-
"""
Copyright 2017 Egor Kobylkin 
Created on Sun Apr 23 11:52:59 2017
@author: ekobylkin
This is an example on how to prepare data for autosklearn-zeroconf.
It is using a well known Adult (Salary) dataset from UCI https://archive.ics.uci.edu/ml/datasets/Adult .
"""
import pandas as pd
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
col_names=[
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

train = pd.read_csv(filepath_or_buffer='../data/adult.data',sep=',', error_bad_lines=False, index_col=False, names=col_names)
category_mapping={' >50K':1,' <=50K':0}
train['category']= train['category'].map(category_mapping)
#dataframe=train

test = pd.read_csv(filepath_or_buffer='../data/adult.test',sep=',', error_bad_lines=False, index_col=False, names=col_names, skiprows=1)
test['set_name']='test'
category_mapping={' >50K.':1,' <=50K.':0}
test['category']= test['category'].map(category_mapping) 

dataframe=train.append(test)

# autosklearn-zeroconf requires cust_id and category (target or "y" variable) columns, the rest is optional
dataframe['cust_id']=dataframe.index

# let's save the test with the cus_id and binarized category for the validation of the prediction afterwards
test_df=dataframe.loc[dataframe['set_name']=='test'].drop(['set_name'], axis=1)
test_df.to_csv('../data/adult.test.withid', index=False, header=True)

# We will use the test.csv data to make a prediction. You can compare the predicted values with the ground truth yourself.
dataframe.loc[dataframe['set_name']=='test','category']=None
dataframe=dataframe.drop(['set_name'], axis=1)

print(dataframe)

store = pd.HDFStore('../data/Adult.h5') # this is the file cache for the data
store['data'] = dataframe
store.close()
#Now run 'python zeroconf.py Adult.h5' (python >=3.5)
