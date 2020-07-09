#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


print('Tensorflow Version: {}'.format(tf.__version__))


# In[4]:


import pandas as pd


# In[5]:


data = pd.read_csv('./dataset/Income1.csv')


# In[6]:


data


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


plt.scatter(data.Education, data.Income)


# In[9]:


x = data.Education
y = data.Income


# In[10]:


model = tf.keras.Sequential()


# In[11]:


model.add(tf.keras.layers.Dense(1, input_shape=(1,)))


# In[12]:


model.summary()   # ax + b


# In[13]:


model.compile(optimizer='adam',
              loss='mse'
)


# In[14]:


history = model.fit(x, y, epochs=5000)


# In[15]:


model.predict(x)


# In[18]:


model.predict(pd.Series([20]))


# In[ ]:




