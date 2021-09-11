#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data processing
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sb

# For labelling
from sklearn.preprocessing import LabelEncoder

# For splitting dataset in for training and testing
from sklearn.model_selection import train_test_split

# Accuracy score
from sklearn.metrics import accuracy_score


# In[5]:


dataset=pd.read_csv("C:\\Users\\PAKISTAN\\Desktop\\Iris.csv")


# In[6]:


dataset.head()


# In[7]:


dataset = dataset.drop(columns=['Id'])
dataset.head()


# In[8]:


dataset.info()


# In[37]:


dataset.head()


# In[38]:


dataset.head(150)


# In[41]:


dataset.describe()


# In[42]:


dataset['class'].value_counts()


# In[9]:


dataset.isnull().sum()


# In[12]:


#Visualisation
import seaborn as sns
sns.set_palette('Set2')

b = sns.pairplot(data = dataset,hue="Species");
plt.show()


# In[14]:


#labeling
label = LabelEncoder()
dataset['Species'] = label.fit_transform(dataset['Species'])
dataset.head()


# In[15]:


dataset.tail()


# In[16]:


#Traing the model
X = dataset.drop(columns=['Species'])
Y = dataset['Species']

# 70% of dataset will we used for training and rest 30% will be used for testing 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


# In[18]:


# Appying decision tree algorithm on our Dataset

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model = model.fit(x_train, y_train)


# In[19]:



print("Accuracy: ", model.score(x_test, y_test)*100)


# In[20]:


#vizualizing  Decission Tree Graph
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model,
               feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
               class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
               filled = True,
              rounded = True)


# In[ ]:




