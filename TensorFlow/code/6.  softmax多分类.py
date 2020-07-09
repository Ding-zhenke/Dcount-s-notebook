#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


print('Tensorflow Version: {}'.format(tf.__version__))


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(train_image, train_lable), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()


# In[5]:


train_image.shape


# In[6]:


train_lable.shape


# In[7]:


test_image.shape, test_label.shape


# In[8]:


plt.imshow(train_image[0])


# In[9]:


np.max(train_image[0])


# In[10]:


train_lable


# In[11]:


train_image = train_image/255
test_image = test_image/255


# In[12]:


train_image.shape


# In[15]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[16]:


model.summary()


# In[20]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[21]:


model.fit(train_image, train_lable, epochs=5)


# In[23]:


model.evaluate(test_image, test_label)


# In[29]:


train_lable


# beijing [1, 0, 0]
# shanghai [0, 1, 0]
# shenzhen  [0, 0, 1]

# In[21]:


train_label_onehot = tf.keras.utils.to_categorical(train_lable)


# In[22]:


train_label_onehot[-1]


# In[23]:


test_label


# In[24]:


test_label_onehot = tf.keras.utils.to_categorical(test_label)


# In[35]:


test_label_onehot


# In[37]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[49]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['acc']
)


# In[50]:


model.fit(train_image, train_label_onehot, epochs=5)


# In[43]:


predict = model.predict(test_image)


# In[45]:


test_image.shape


# In[44]:


predict.shape


# In[46]:


predict[0]


# In[47]:


np.argmax(predict[0])


# In[48]:


test_label[0]


# In[45]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[46]:


model.summary()


# In[47]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
)


# In[48]:


history = model.fit(train_image, train_label_onehot, 
                    epochs=10, 
                    validation_data=(test_image, test_label_onehot))


# In[41]:


history.history.keys()


# In[49]:


plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()


# In[50]:


plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()


# In[ ]:





# In[43]:


plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()


# In[44]:


plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()


# 过拟合： 在训练数据上得分很高， 在测试数据上得分相对比较低

# 欠拟合：  在训练数据上得分比较低， 在测试数据上得分相对比较低

# In[51]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[52]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
)


# In[53]:


history = model.fit(train_image, train_label_onehot, 
                    epochs=10, 
                    validation_data=(test_image, test_label_onehot))


# In[54]:


plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()


# In[ ]:




