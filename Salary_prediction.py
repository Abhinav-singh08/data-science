import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split   

df= pd.read_csv('Salary.csv')
df
x= df.iloc[:, : -1].values
y= df.iloc[:, 1].values
y
x
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size= 1/4, random_state= 0)
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color= 'red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('training set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show 

plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_test, regressor.predict(x_test),color='blue')
plt.title('test set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show
