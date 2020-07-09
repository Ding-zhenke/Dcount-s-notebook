#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


print('Tensorflow version: {}'.format(tf.__version__))


# In[3]:


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7])


# In[4]:


dataset


# In[6]:


for ele in dataset:
    print(ele.numpy())


# In[8]:


dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])


# In[9]:


dataset


# In[10]:


for ele in dataset:
    print(ele.numpy())


# In[11]:


dataset_dic = tf.data.Dataset.from_tensor_slices({'a': [1,2,3,4],
                                                  'b': [6,7,8,9],
                                                  'c': [12,13,14,15]
    
})


# In[13]:


dataset_dic


# In[14]:


for ele in dataset_dic:
    print(ele)


# In[15]:


import numpy as np


# In[16]:


dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6, 7]))


# In[19]:


for ele in dataset.take(4):
    print(ele.numpy())


# In[22]:


next(iter(dataset.take(1)))


# In[23]:


dataset


# In[31]:


dataset = dataset.shuffle(7)
dataset = dataset.repeat()
dataset = dataset.batch(3)


# In[32]:


for ele in dataset:
    print(ele.numpy())


# In[33]:


dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6, 7]))


# In[34]:


dataset = dataset.map(tf.square)


# In[35]:


for ele in dataset:
    print(ele.numpy())


# In[ ]:




