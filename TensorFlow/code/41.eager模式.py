#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


tf.executing_eagerly()


# In[4]:


x = [[2,]]


# In[5]:


m = tf.matmul(x, x)


# In[6]:


print(m)


# In[7]:


m.numpy()


# In[9]:


a = tf.constant([[1, 2],
                [3, 4]])


# In[10]:


a


# In[11]:


a.numpy()


# In[12]:


b = tf.add(a, 1)


# In[13]:


b


# In[19]:


a


# In[20]:


import numpy as np


# In[21]:


d = np.array([[5,6], 
              [7, 8]])


# In[22]:


d


# In[24]:


(a + b).numpy()


# In[26]:


g = tf.convert_to_tensor(10)


# In[27]:


g


# In[28]:


float(g)


# In[14]:


c = tf.multiply(a, b)


# In[15]:


c


# In[16]:


num = tf.convert_to_tensor(10)


# In[17]:


num


# In[18]:


for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i%2) == 0:
        print('even')
    else:
        print('odd')


# In[ ]:




