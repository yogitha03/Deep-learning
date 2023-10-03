#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


data = pd.read_csv("D:/downloads/archive/diabetes.csv")


# In[3]:


data.head(5)


# In[4]:


import seaborn as sns
data['Outcome'].value_counts().plot(kind='bar')


# In[5]:


predictors = data.iloc[:,0:8]
response = data.iloc[:,8]


# In[6]:


x_train,x_test,y_train,y_test = train_test_split(predictors,response,test_size = 0.2)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[7]:


kerasmodel=Sequential()
kerasmodel.add(Dense(12, input_dim=8, activation='relu'))
kerasmodel.add(Dense(8, activation='relu'))
kerasmodel.add(Dense(1,activation='sigmoid'))


# In[8]:


kerasmodel.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[9]:


kerasmodel.fit(x_train,y_train,epochs=150,batch_size = 10)


# In[10]:


_, accuracy = kerasmodel.evaluate(x_train,y_train)
print('Train accuracy: %2f' %(accuracy*100))


# In[ ]:




