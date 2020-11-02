#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.utils import resample


# In[17]:


corona_dataset_url = "https://docs.google.com/uc?id=1CA1RPRYqU9oTIaHfSroitnWrI6WpUeBw&export=download"
urlRequest = urllib.request.Request(corona_dataset_url)
datasetFile = urllib.request.urlopen(urlRequest)


# In[18]:


corona_dataset = pd.read_csv(datasetFile)


# In[19]:


corona_dataset.head()


# In[20]:


corona_dataset['sex'] = pd.factorize(corona_dataset['sex'])[0]
corona_dataset['country'] = pd.factorize(corona_dataset['country'])[0]
corona_train = corona_dataset.drop("deceased", axis=1)


# In[21]:


corona_dataset.head()


# In[22]:


len(corona_dataset)


# In[23]:


len(corona_dataset[corona_dataset['deceased'] == 1])*100/len(corona_dataset)


# As we can see the dataset is unbalanced. There is only 2% of positive case for 98% negative case in the database. 

# In[53]:


def getOptimalCluster():
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(corona_train)
        Sum_of_squared_distances.append(km.inertia_)
    print("Sum of squared distances")
    print(Sum_of_squared_distances)
    plt.plot(np.linspace(1,14,14), Sum_of_squared_distances)
    plt.xlabel("Number of cluster chosen")
    plt.ylabel('Sum squared distances')
    plt.show()


# In[54]:


getOptimalCluster()


# In[71]:


def launchKmeans(nbrOfClusters, true_labels):
    km = KMeans(n_clusters=2)
    predicted_labels = km.fit_predict(corona_train)
    print("Accuracy : " + str(accuracy_score(true_labels, predicted_labels)))
    print("Silhouette coefficient : " + str(silhouette_score(corona_train, predicted_labels)))
    print("Silhouette coefficient : " + str(davies_bouldin_score(corona_train, predicted_labels)))


# In[72]:


launchKmeans(nbrOfClusters=2, true_labels=corona_dataset.deceased)


# We can see that because the dataset is not balanced the accuracy score is low.

# The Dun index shows as well that the two clusters are too close

# In[61]:


df_majority = corona_dataset[corona_dataset.deceased==0]
df_minority = corona_dataset[corona_dataset.deceased==1]
 
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=45,    
                                 random_state=123) 
 
corona_dataset_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
corona_dataset_downsampled.deceased.value_counts()


# In[62]:


corona_train = corona_dataset_downsampled.drop('deceased', axis=1)


# In[63]:


getOptimalCluster()


# In[74]:


launchKmeans(nbrOfClusters=2, true_labels=corona_dataset_downsampled.deceased)


# In[70]:


x = np.linspace(0,9,90)
plt.scatter(x, corona_dataset_downsampled.age, c=corona_dataset_downsampled.deceased)
plt.show()

