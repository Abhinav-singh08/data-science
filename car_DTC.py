#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:36:16 2018

@author: abhinav
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns=['buying',' maint',' doors',' persons',' lug_boot', 'safety', 'class value']
df= pd.read_csv('car.data', header= None, names= columns)
df
x= df.iloc[:,: -1].values
y= df.iloc[:, -1].values
x
y
from sklearn.preprocessing import LabelEncoder
labelencoder_x= LabelEncoder()
labelencoder_y= LabelEncoder()
x[:,0]= labelencoder_x.fit_transform(x[:,0])
x[:,1]= labelencoder_x.fit_transform(x[:,1])
x[:,4]= labelencoder_x.fit_transform(x[:,4])
x[:,5]= labelencoder_x.fit_transform(x[:,5])
x[:,2]= labelencoder_x.fit_transform(x[:,2])
x[:,3]= labelencoder_x.fit_transform(x[:,3])

x
y

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 1/4, random_state=0)
x

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(random_state=0)
classifier.fit(x_train,y_train)
test= [[1,1,1,1,1,1]]
y_pred= classifier.predict(test)
y_pred















