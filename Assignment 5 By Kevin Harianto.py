#!/usr/bin/env python
# coding: utf-8

# 1- Load the Android Malware dataset into a Pandas DataFrame.

# In[23]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
app_df= pd.read_pickle('E:/downld/Android_malware.pkl')


# 2- Select columns 8 to end as features (X).

# In[24]:


from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Split the data into features (X) and class (y)
X = app_df.iloc[:, 8:]


# 3- Defines a range of K values to try (from 2 to 20).

# In[25]:


# Define the range of K values to try
k_values = range(2, 20)


# 4- Initializes a list to store the WCSS values for different K.

# In[26]:


wcss_values = {}
inertias = []


# 5- Iterates over the range of K values:
#     5-1- Creates a K-means clustering object for each K.
#     5-2-Fits the K-means model to the data.
#     5-3- Calculates the WCSS (within-cluster sum of squares) for each K.
#     5-4- Stores the WCSS value for each K.

# In[27]:


# Iterate over different K values
for k in k_values:
    # Create a k-means clustering object
    kmeans = KMeans(n_clusters=k)

    # Fit the k-means model to the data
    kmeans.fit(X)
    
    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_
    
    # Calculate the WCSS (within-cluster sum of squares)
    wcss = np.sum(np.square(X - kmeans.cluster_centers_[cluster_labels]), axis=1)
    
    # Store the WCSS value for this K
    wcss_values[k] = np.sum(wcss)


# In[30]:


from sklearn.preprocessing import StandardScaler
k_values = list(range(2,20))
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)
for k in k_values:
    km=KMeans(n_clusters=k)
    km.fit(scaled_df)
    inertias.append(km.inertia_)


# 6- Plots the elbow chart showing the WCSS against the number of clusters (K).

# In[31]:


import matplotlib.pyplot as plt
plt.plot(k_values, inertias, marker='.')
plt.xticks(k_values)
plt.xlabel('number of clusters')
plt.ylabel('WCSS')


# 7- Identifies the best K by visually inspecting the elbow plot.

# Since the lower the WCSS is better, the ideal value would be the lowest WCSS value. Which in this case would be k = 19

# In[32]:


# Find the best K with the lowest WCSS value
best_k = min(wcss_values, key=wcss_values.get)

print("Best K:", best_k)
print("WCSS:", wcss_values[best_k])


# This is further proven when running the code that determines the lowest k value

# In[ ]:




