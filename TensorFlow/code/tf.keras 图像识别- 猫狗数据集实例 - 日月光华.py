#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob


# In[2]:


tf.__version__


# In[3]:


image_filenames = glob.glob('./dc/train/*.jpg')


# In[26]:


len(image_filenames)


# In[27]:


25000/32


# In[4]:


image_filenames = np.random.permutation(image_filenames)


# In[5]:


lables = list(map(lambda x: float(x.split('\\')[1].split('.')[0] == 'cat'), image_filenames))


# In[6]:


dataset = tf.data.Dataset.from_tensor_slices((image_filenames, lables))    #创建dataset


# In[7]:


dataset


# In[8]:


def _pre_read(img_filename, lable):
    image = tf.read_file(img_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,(200,200))
    image = tf.reshape(image,[200,200,1])
    image = tf.image.per_image_standardization(image)
    lable = tf.reshape(lable, [1])
    return image, lable


# In[9]:


dataset = dataset.map(_pre_read)      # 转换


# In[10]:


dataset = dataset.shuffle(300)


# In[11]:


dataset = dataset.repeat()


# In[12]:


dataset = dataset.batch(32)


# In[13]:


dataset


# 创建模型   #binary_crossentropy

# In[14]:


model = keras.Sequential()


# In[15]:


model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 1)))


# In[16]:


model.add(layers.MaxPooling2D((2, 2)))


# In[17]:


model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[18]:


model.add(layers.MaxPooling2D((2, 2)))


# In[19]:


model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[20]:


model.add(layers.MaxPooling2D((2, 2)))


# In[21]:


model.add(layers.Flatten())


# In[22]:


model.add(layers.Dense(64, activation='relu'))


# In[23]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[24]:


model.summary()


# In[25]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[28]:


model.fit(dataset, epochs=10, steps_per_epoch=781, validation_data=dataset_test, validation_steps=781)


# In[ ]:




