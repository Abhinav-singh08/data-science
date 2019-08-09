#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 12:03:02 2019

@author: abhinav
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as kr
import tensorflow as ts


dataset_train = pd.read_csv("criminal_train.csv")
X = dataset_train.iloc[:, 1:71].values
y = dataset_train.iloc[:, 71].values
dataset_test = pd.read_csv("criminal_test.csv")
z = dataset_test.iloc[:, 1:71]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
z = sc.transform(z)




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()


classifier.add(Dense(units = 150, kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y,  epochs = 3)
 


y_pred2 = classifier.predict(z)
y_pred2 = (y_pred2 > 0.5)




