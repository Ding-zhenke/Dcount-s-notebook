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


train_images.shape


# In[6]:


train_labels.shape


# In[7]:


test_images.shape


# In[8]:


test_labels


# In[9]:


plt.imshow(train_images[0])


# In[10]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[11]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[13]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


# In[14]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


history = model.fit(train_images, 
                    train_labels, 
                    epochs=30, 
                    validation_data=(test_images, test_labels))


# In[16]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[17]:


plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
plt.plot(history.epoch, history.history['val_loss'], 'b--', label='val_loss')
plt.legend()


# In[18]:


plt.plot(history.epoch, history.history['accuracy'], 'r')
plt.plot(history.epoch, history.history['val_accuracy'], 'b--')


# In[19]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


# In[20]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[21]:


history = model.fit(train_images, 
                    train_labels, 
                    epochs=30, 
                    validation_data=(test_images, test_labels))


# In[22]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[23]:


plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
plt.plot(history.epoch, history.history['val_loss'], 'b--', label='val_loss')
plt.legend()


# In[24]:


plt.plot(history.epoch, history.history['accuracy'], 'r')
plt.plot(history.epoch, history.history['val_accuracy'], 'b--')


# In[ ]:




