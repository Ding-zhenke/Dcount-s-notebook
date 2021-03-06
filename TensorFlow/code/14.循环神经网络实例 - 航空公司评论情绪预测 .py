#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('./dataset/Tweets.csv')


# In[3]:


data.head()


# In[4]:


data = data[['airline_sentiment', 'text']]


# In[5]:


data.airline_sentiment.unique()


# In[6]:


data.airline_sentiment.value_counts()


# In[7]:


data_p = data[data.airline_sentiment == 'positive']


# In[8]:


data_n = data[data.airline_sentiment == 'negative']


# In[9]:


data_n = data_n.iloc[:len(data_p)]


# In[10]:


len(data_n), len(data_p)


# In[11]:


data = pd.concat([data_n, data_p])


# In[12]:


data = data.sample(len(data))


# In[13]:


data['review'] = (data.airline_sentiment == 'positive').astype('int')


# In[14]:


del data['airline_sentiment']


# tf.keras.layers.Embedding  把文本向量化

# In[15]:


import re


# In[16]:


token = re.compile('[A-Za-z]+|[!?,.()]')


# In[17]:


def reg_text(text):
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text


# In[18]:


data['text'] = data.text.apply(reg_text)


# In[19]:


word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word) 


# In[20]:


max_word = len(word_set) + 1
max_word


# In[21]:


word_list = list(word_set)


# In[22]:


word_list.index('spending')


# In[23]:


word_index =  dict((word, word_list.index(word) + 1) for word in word_list)


# In[24]:


word_index


# In[25]:


data_ok = data.text.apply(lambda x: [word_index.get(word, 0) for word in x])


# In[26]:


len(data_ok.iloc[2])


# In[27]:


maxlen = max(len(x) for x in data_ok)


# In[28]:


maxlen


# In[29]:


data_ok = keras.preprocessing.sequence.pad_sequences(data_ok.values, maxlen=maxlen)


# In[30]:


data_ok.shape


# In[31]:


data.review.values


# In[32]:


model = keras.Sequential()


# Embeding : 把文本映射为一个密集向量

# In[33]:


model.add(layers.Embedding(max_word, 50, input_length=maxlen))


# In[34]:


model.add(layers.LSTM(64))


# In[35]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[36]:


model.summary()


# In[37]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[38]:


history = model.fit(data_ok, data.review.values, epochs=10, batch_size=128, validation_split=0.2)


# In[38]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()


# In[42]:


plt.plot(history.epoch, history.history.get('val_loss'), c='r', label='val_loss')
plt.plot(history.epoch, history.history.get('loss'), c='b', label='loss')
plt.legend()


# 使用循环 dropout 抑制过拟合

# 对每一个时间步使用相同的 dropout 掩码

# In[63]:


def train_model():
    model = keras.Sequential()
    model.add(layers.Embedding(max_word, 16, input_length=maxlen))
    model.add(layers.LSTM(64,
                         dropout=0.2,
                         recurrent_dropout=0.5))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])
    return model


# In[64]:


model2 = train_model()


# In[65]:


history = model2.fit(data_ok, data.review.values, epochs=10, batch_size=128, validation_split=0.2)


# In[66]:


plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()


# In[67]:


plt.plot(history.epoch, history.history.get('val_loss'), c='r', label='val_loss')
plt.plot(history.epoch, history.history.get('loss'), c='b', label='loss')
plt.legend()


# # 使用双向 RNN

# In[78]:


def train_model():
    model = keras.Sequential()
    model.add(layers.Embedding(max_word, 16, input_length=maxlen))
    model.add(layers.Bidirectional(layers.LSTM(64,
                         dropout=0.2,
                         recurrent_dropout=0.5)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])
    return model


# In[79]:


model3 = train_model()


# In[80]:


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=0.00001)


# In[81]:


history = model3.fit(data_ok, 
                     data.review.values, 
                     epochs=30, 
                     batch_size=128, 
                     validation_split=0.2,
                     callbacks=[learning_rate_reduction])


# In[82]:


plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()


# In[83]:


plt.plot(history.epoch, history.history.get('val_loss'), c='r', label='val_loss')
plt.plot(history.epoch, history.history.get('loss'), c='b', label='loss')
plt.legend()


# In[ ]:




