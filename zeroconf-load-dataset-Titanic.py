# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 17:13:59 2016

@author: ekobylkin
"""
#https://www.kaggle.com/overratedgman/breast-cancer-wisconsin-data
import pandas as pd

train = pd.read_csv(filepath_or_buffer='train.csv',sep=',', error_bad_lines=False, index_col=False)
test = pd.read_csv(filepath_or_buffer='test.csv',sep=',', error_bad_lines=False, index_col=False)
test['Survived']=None
dataframe=train.append(test)
dataframe.rename(columns = {'PassengerId':'cust_id','Survived':'category'},inplace=True)
dataframe['category']=dataframe['category'].astype('category').cat.codes #None->-1, 0->0,1->1

X_pred= dataframe[ dataframe.category == -1 ]

#dataframe['category']=dataframe['category'].apply(lambda x : 1 if x>0 else 0)
print(dataframe.dtypes)
store = pd.HDFStore('Titanic.h5') # this is the file cache for the data
store['data'] = dataframe
store.close()

