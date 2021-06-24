#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
import neptune.new as neptune
from sklearn.model_selection import train_test_split


# In[10]:


run = neptune.init(project='projektmlcdv/ml-project-MW-MM')


# ##### Loading data

# In[3]:


encoder = lambda x:1 if x == -1 else 0
decoder = lambda x:-1 if x == 1 else 1


# In[4]:


y = pd.read_csv('train_labels.csv', header=None, names=['y'])
X = pd.read_csv("train_data.csv",header=None, low_memory = False) 


# In[5]:


y= y['y'].apply(encoder)


# ##### Splitting data to train and test sets

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y)


# ##### Dummy classifier

# In[11]:


model = DummyClassifier(strategy = 'stratified')
base_dummy = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report_imbalanced(y_test, y_pred))


# ##### Decision Tree Classifier

# In[12]:


model = DecisionTreeClassifier()
base_dummy = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report_imbalanced(y_test, y_pred))

