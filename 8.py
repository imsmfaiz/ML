import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

data = pd.read_csv('ex.csv')

f1 = data['V1'].values
f2 = data['V2'].values
x = np.array(list(zip(f1, f2)))
print("x: ", x)

print('Graph for whole dataset')
plt.scatter(f1, f2, c='black')  
plt.show()

kmeans=KMeans(2)
labels =kmeans.fit(x).predict(x)
print("labels for kmeans:", labels)

print('Graph using Kmeans Algorithm')
plt.scatter(f1, f2, c=labels)
plt.show()

centroids = kmeans.cluster_centers_
print("centroids:", centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red')
plt.show()

labels = GaussianMixture(2).fit(x).predict(x)
print("Labels for GMM: ", labels)

print('Graph using EM Algorithm')
plt.scatter(f1, f2, c=labels)
plt.show()

