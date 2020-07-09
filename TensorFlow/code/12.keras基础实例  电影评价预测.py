#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras import layers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = keras.datasets.imdb


# In[3]:


max_word = 10000


# In[4]:


(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)


# In[5]:


x_train.shape, y_train.shape


# In[11]:


x_train[0]


# In[7]:


y_train


# In[8]:


word_index = data.get_word_index()


# In[9]:


index_word = dict((value, key) for key,value in word_index.items())


# In[10]:


[index_word.get(index-3, '?') for index in x_train[0]]


# In[12]:


[len(seq) for seq in x_train]


# In[15]:


max([max(seq) for seq in x_train])


# 文本的向量化

# one-hot

# k-hot

# In[22]:


import numpy as np


# In[25]:


def k_hot(seqs, dim=10000):
    result = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        result[i, seq] = 1
    return result


# In[26]:


x_train = k_hot(x_train)


# In[27]:


x_train.shape


# In[29]:


x_train[0].shape


# In[30]:


x_test = k_hot(x_test)


# In[32]:


y_train


# In[50]:


model = keras.Sequential()


# In[51]:


model.add(layers.Dense(32, input_dim=10000, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[52]:


model.summary()


# In[53]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[54]:


history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))


# In[57]:


plt.plot(history.epoch, history.history.get('loss'), c='r', label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), c='b', label='val_loss')
plt.legend()


# In[59]:


plt.plot(history.epoch, history.history.get('acc'), c='r', label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), c='b', label='val_acc')
plt.legend()


# In[ ]:


keras.datasets.

