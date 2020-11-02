#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.metrics import silhouette_score, davies_bouldin_score


# In[18]:


gene_dataset_url = "https://docs.google.com/uc?id=1VfVCQvWt121UN39NXZ4aR9Dmsbj-p9OU&export=download"
urlRequest = urllib.request.Request(gene_dataset_url)
datasetFile = urllib.request.urlopen(urlRequest)


# In[19]:


tissu_dataset = pd.read_csv(datasetFile, header=None)


# In[20]:


trans_dataset = pd.DataFrame()
for i in range(0, len(tissu_dataset)):
    trans_dataset[i] = tissu_dataset.iloc[i]
tissu_dataset = trans_dataset


# In[21]:


tissu_dataset.head()


# In[22]:


len(tissu_dataset)


# In[23]:


tissu_dataset.isnull().values.any()


# In[24]:


tissu_dataset.info()


# In[45]:


pca = PCA(n_components=40)
pca.fit(tissu_dataset)


# In[46]:


plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel('Principal Component')
plt.ylabel('CPVE')
plt.show()


# In[28]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(tissu_dataset)
    Sum_of_squared_distances.append(km.inertia_)


# In[29]:


Sum_of_squared_distances


# In[41]:


plt.plot(np.linspace(1,14,14), Sum_of_squared_distances)
plt.xlabel('Sum squared distances')
plt.ylabel('Number of clusters')
plt.title("Sum of the squared distances depending on the number of clusters")
plt.show()


# In[31]:


km = KMeans(n_clusters=2)
km = km.fit_predict(tissu_dataset)


# In[32]:


silhouette_score(tissu_dataset, km)


# The silhouette coefficient estimates the average distance between clusters. If the coefficient tends toward 1 then it means that the values are well clustered. In that case it means that a lot of values 

# In[33]:


davies_bouldin_score(tissu_dataset, km)


# According to the silhouette cofficient and the Dunn Index it seams that the two clusters are a bit too close

# In[34]:


agglo = FeatureAgglomeration(n_clusters=2)
agglo.fit(tissu_dataset)


# In[35]:


tissu_reduced = agglo.transform(tissu_dataset)
tissu_reduced.shape


# In[36]:


km = KMeans(n_clusters=2)
km = km.fit(tissu_reduced)


# In[42]:


plt.plot(tissu_reduced[0:20,0],tissu_reduced[0:20,1], 'ro', marker = "+", color='g')
plt.plot(tissu_reduced[21:40,0], tissu_reduced[21:40,1], 'ro', marker = "X",color='r')
plt.plot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], 'ro', marker = "X",color='b')
plt.title("Plotting the two clusters with their center depending on their label class")
plt.show()


# In[38]:


km.labels_


# In[43]:


plt.scatter(tissu_reduced[:,0], tissu_reduced[:,1], c=km.labels_)
plt.plot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], 'ro', marker = "X",color='r')
plt.title("Plotting the two clusters with their center depending on their label class predicted by kMeans")
plt.show()

