#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


dp=pd.read_csv("C:/Users/shrut/Downloads/homeprices.csv")


# In[5]:


dp


# In[6]:


dp.head()


# In[7]:


dp.tail()


# In[8]:


dp.info()


# In[10]:


dp.plot()


# In[11]:


dp.plot(kind="hist",color="red")


# In[20]:


dp.plot(kind="scatter",marker="*",x="area",y="price",color="red")


# In[26]:


dp.plot(x="area",y="price",marker='*',linestyle="dotted",color="red",ms=20,mfc='b')


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[28]:


X=dp[['area']]
Y=dp[['price']]


# In[69]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)


# In[70]:


X_train


# In[71]:


X_test


# In[72]:


Y_test


# In[73]:


Y_train


# In[74]:


reg=LinearRegression()
reg.fit(X_train,Y_train)


# In[75]:


reg.predict(X_test)


# In[76]:


reg.score(X_test,Y_test)


# In[ ]:




