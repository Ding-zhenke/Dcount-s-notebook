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


# In[7]:


data = pd.read_csv('dataset/credit-a.csv', header=None)


# In[8]:


data.head()


# In[9]:


data.iloc[:, -1].value_counts()


# In[10]:


x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)


# In[11]:


model = tf.keras.Sequential()


# In[12]:


model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[13]:


model.summary()


# In[14]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[15]:


history = model.fit(x, y, epochs=100)


# In[16]:


history.history.keys()


# In[17]:


plt.plot(history.epoch, history.history.get('loss'))


# In[18]:


plt.plot(history.epoch, history.history.get('acc'))


# In[ ]:




