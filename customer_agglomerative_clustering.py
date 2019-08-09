import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('Customers.csv')

x= df.iloc[:, [3,4]].values
import scipy.cluster.hierarchy as sch
dendogram= sch.dendrogram(sch.linkage(x, method= 'ward'))

plt.title('dendrone')
plt.xlabel('customer')
plt.ylabel('euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
sch=  AgglomerativeClustering(n_clusters= 5, affinity= 'euclidean', linkage= 'ward')
y_sch= sch.fit_predict(x)

plt.scatter(x[y_sch== 0,0],x[y_sch== 0,1], s=100,c= 'red', label= 'cluster1')
plt.scatter(x[y_sch== 1,0],x[y_sch== 1,1], s=100,c= 'blue', label= 'cluster2')
plt.scatter(x[y_sch== 2,0],x[y_sch== 2,1], s=100,c= 'green', label= 'cluster3')
plt.scatter(x[y_sch== 3,0],x[y_sch== 3,1], s=100,c= 'black', label= 'cluster4')
plt.scatter(x[y_sch== 4,0],x[y_sch== 4,1], s=100,c= 'cyan', label= 'cluster5')

plt.scatter(y_sch.cluster_centers_[:,0], y_sch.cluster_centers_[:, 1], s= 200, c= 'yellow', label= 'centroids')
plt.title('cluster of customes')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()