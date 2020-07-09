#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import glob
import os


# In[2]:


print('Tensorflow version: {}'.format(tf.__version__))


# In[3]:


tf.test.is_gpu_available()


# In[4]:


keras = tf.keras
layers = tf.keras.layers


# In[5]:


train_image_path = glob.glob('./dc/train/*/*.jpg')


# In[6]:


len(train_image_path)


# In[7]:


train_image_path[-5:]


# In[8]:


train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]


# In[9]:


train_image_label[-5:]


# In[10]:


train_image_label[ :5]


# In[11]:


def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label


# In[12]:


train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))


# In[13]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[14]:


train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)


# In[15]:


train_image_ds


# In[16]:


for img, label in train_image_ds.take(2):
    plt.imshow(img)


# In[17]:


BATCH_SIZE = 32
train_count = len(train_image_path)


# In[18]:


train_image_ds = train_image_ds.shuffle(train_count).repeat().batch(BATCH_SIZE)


# In[19]:


test_image_path = glob.glob('./dc/test/*/*.jpg')
test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.repeat().batch(BATCH_SIZE)


# In[20]:


test_count = len(test_image_path)
test_count


# keras内置经典网络实现

# In[21]:


covn_base = keras.applications.xception.Xception(weights='imagenet', 
                                                 include_top=False,
                                                 input_shape=(256, 256, 3),
                                                 pooling='avg')


# In[22]:


covn_base.trainable = False


# In[23]:


covn_base.summary()


# In[24]:


model = keras.Sequential()
model.add(covn_base)
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[25]:


model.summary()


# In[26]:


model.summary()


# In[27]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['acc'])


# In[35]:


initial_epochs = 5


# In[29]:


history = model.fit(
    train_image_ds,
    steps_per_epoch=train_count//BATCH_SIZE,
    epochs=initial_epochs,
    validation_data=test_image_ds,
    validation_steps=test_count//BATCH_SIZE)


# In[30]:


covn_base.trainable = True


# In[31]:


len(covn_base.layers)


# In[32]:


fine_tune_at = -33


# In[33]:


for layer in covn_base.layers[:fine_tune_at]:
    layer.trainable =  False


# In[34]:


model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.Adam(lr=0.0005/10),
              metrics=['accuracy'])


# In[36]:


fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs


history = model.fit(
    train_image_ds,
    steps_per_epoch=train_count//BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch = initial_epochs,
    validation_data=test_image_ds,
    validation_steps=test_count//BATCH_SIZE)


# In[ ]:





# 
