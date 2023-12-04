#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


data=pd.DataFrame(dict(yearsexperience=[1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5],salary=[39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872]))


# In[5]:


data.head(5)


# In[6]:


sns.scatterplot(x="yearsexperience",y="salary",data=data)


# #### Objective -  Is to build a prediction model for Salary Hike.

# In[8]:


x=data["yearsexperience"].values.reshape(-1,1)


# In[9]:


y=data["salary"].values.reshape(-1,1)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[12]:


lin=LinearRegression()


# In[14]:


lin.fit(x,y)
print("training completed")


# In[16]:


plt.scatter(x,y,color="b")
plt.plot(x,lin.predict(x),color="r")


# In[17]:


from sklearn.metrics import r2_score


# In[18]:


r2_score(y,lin.predict(x))


# In[21]:


print("r2_score =",r2_score(y,lin.predict(x)),"\nvery strong model")


# In[ ]:




