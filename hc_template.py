# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 00:45:53 2019

@author: sagar
"""

# HC Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values
import scipy.sparse
import scipy.cluster.hierarchy  as sch

z = sch.linkage(X, method = 'ward')
dendrogram = sch.dendrogram(z)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()