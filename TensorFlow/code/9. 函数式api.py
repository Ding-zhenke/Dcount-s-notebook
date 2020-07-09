#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


print('Tensorflow version: {}'.format(tf.__version__))


# In[3]:


from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[5]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[6]:


train_images.shape


# In[17]:


input1 = keras.Input(shape=(28, 28))


# In[18]:


input2 = keras.Input(shape=(28, 28))


# In[19]:


x1 = keras.layers.Flatten()(input1)
x2 = keras.layers.Flatten()(input2)


# In[20]:


x = keras.layers.concatenate([x1, x2])


# In[21]:


x = keras.layers.Dense(32, activation='relu')(x)


# In[22]:


output = keras.layers.Dense(1, activation='sigmoid')(x)


# In[23]:


model = keras.Model(inputs=[input1, input2], outputs=output)


# In[24]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


x = keras.layers.Dense(32, activation='relu')(x)


# In[10]:


x = keras.layers.Dropout(0.5)(x)


# In[11]:


x = keras.layers.Dense(64, activation='relu')(x)


# In[12]:


output = keras.layers.Dense(10, activation='softmax')(x)


# In[13]:


model = keras.Model(inputs=input, outputs=output)


# In[14]:


model.summary()


# In[15]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


history = model.fit(train_images, 
                    train_labels, 
                    epochs=30, 
                    validation_data=(test_images, test_labels))


# In[24]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[25]:


plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
plt.plot(history.epoch, history.history['val_loss'], 'b--', label='val_loss')
plt.legend()


# In[26]:


plt.plot(history.epoch, history.history['accuracy'], 'r')
plt.plot(history.epoch, history.history['val_accuracy'], 'b--')


# In[ ]:




