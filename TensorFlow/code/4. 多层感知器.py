#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


print('Tensorflow Version: {}'.format(tf.__version__))


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv('dataset/Advertising.csv')


# In[6]:


data.head()


# In[8]:


plt.scatter(data.TV, data.sales)


# In[9]:


plt.scatter(data.radio, data.sales)


# In[10]:


plt.scatter(data.newspaper, data.sales)


# In[11]:


x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]


# In[12]:


model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)]
)


# In[13]:


model.summary()


# In[14]:


model.compile(optimizer='adam',
              loss='mse'
)


# In[15]:


model.fit(x, y, epochs=100)


# In[16]:


test = data.iloc[:10, 1:-1]


# In[17]:


model.predict(test)


# In[18]:


test = data.iloc[:10, -1]


# In[19]:


test


# In[ ]:




