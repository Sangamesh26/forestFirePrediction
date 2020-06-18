#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("forestFire.csv")


# In[3]:


df


# In[4]:


df['Area']=df['Area'].replace(np.nan,"Mysuru")


# In[5]:


df


# In[6]:


data=np.array(df)


# In[7]:


data


# In[10]:


X=data[:,1:-1]
y=data[:,-1]


# In[11]:


X


# In[12]:


y


# In[19]:


X=X.astype('int')
y=y.astype('int')


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[22]:


log_reg=LogisticRegression()


# In[23]:


log_reg.fit(X_train,y_train)


# In[70]:


inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]


# In[71]:


b=log_reg.predict(final)


# In[72]:


b


# In[73]:


b=log_reg.predict_proba(final)


# In[74]:


b


# In[75]:


output='{0:-{1}f}'.format(b[0][1],2)


# In[76]:


output


# In[78]:


score=log_reg.score(X_test,y_test)


# In[79]:


score


# In[ ]:




