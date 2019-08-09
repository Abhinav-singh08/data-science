@author: abhinav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split   

columns = ['s_length','s_ width','p_length','p_width','type']
df= pd.read_csv('iris.data', header=None , names= columns)
df

x= df.iloc[:, : -1].values
y= df.iloc[:, -1].values
y
from sklearn.preprocessing import LabelEncoder
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size= 1/4, random_state= 0)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(random_state= 0)
classifier.fit(x_train, y_train)

type = [[7,2,5,1]]
ypred= classifier.predict(type )
ypred

from sklearn.metrics import accuracy_score
accuracy_score(ypred, y_test)

x_grid= np.arange(min(X), max(X),0.01)
x_grid= X_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color= 'red')
plt.plot(x_grid, classifier.predict(x_grid),color='blue')
plt.title('Decision Tree Classifier')

plt.show()
