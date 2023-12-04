#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


data=pd.DataFrame(dict(delivery_time=[21,13.5,19.75,24,29,15.35,19,9.5,17.9,18.75,19.83,10.75,16.68,11.5,12.03,14.88,13.75,18.11,8,17.83,21.5],sorting_time=[10,4,6,9,10,6,7,3,10,9,8,4,7,3,3,4,6,7,2,7,5]))


# In[28]:


data.head(5)


# In[21]:


sns.scatterplot(x="sorting_time",y="delivery_time",data=data)


# #### Objective  - Is to predict the Delivery time using Sorting time using simple Linear Regression

# In[15]:


x=data["sorting_time"].values.reshape(-1,1)


# In[16]:


y=data["delivery_time"].values.reshape(-1,1)


# In[17]:


from sklearn.linear_model import LinearRegression
lin=LinearRegression()


# In[19]:


lin.fit(x,y)
print("training completed")


# In[23]:


plt.scatter(x,y,color="b")
plt.plot(x,lin.predict(x),color="r")


# In[24]:


from sklearn.metrics import r2_score


# In[25]:


r2_score(y,lin.predict(x))


# In[26]:


print("r2_score is =",r2_score(y,lin.predict(x)))


# In[27]:


print("value of r2_score is closer to 1 the model is strong model")


# In[ ]:




