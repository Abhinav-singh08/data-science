import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 

columns= ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black','lstat', 'medv']
df= pd.read_csv('housing.data', header = None, delim_whitespace= True, names= columns)
df

x= df.iloc[:,: -1].values
y= df.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size= 1/4, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
type= [[0.00632,	18.0,	2.31	,0,	0.538,	6.575,	65.2	,4.09,	1	,296.0,	15.3,	396.9,	4.98]]
y_pred= regressor.predict(type)
y_pred