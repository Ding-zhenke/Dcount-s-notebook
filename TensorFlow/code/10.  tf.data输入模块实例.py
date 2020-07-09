#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


print('Tensorflow version: {}'.format(tf.__version__))


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# In[5]:


train_images.shape


# In[6]:


train_labels.shape


# In[7]:


train_images[0]


# In[8]:


plt.imshow(train_images[0])


# In[9]:


train_images = train_images/255
test_images = test_images/255


# In[10]:


dataset_images = tf.data.Dataset.from_tensor_slices(train_images)


# In[11]:


dataset_images


# In[12]:


dataset_labels = tf.data.Dataset.from_tensor_slices(train_labels)


# In[13]:


dataset_labels


# In[14]:


dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))


# In[15]:


dataset


# In[21]:


batch_size = 256


# In[16]:


dataset = dataset.shuffle(train_images.shape[0]).repeat().batch(batch_size)


# In[18]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[19]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[22]:


steps_per_epoch = train_images.shape[0]/batch_size


# In[23]:


model.fit(dataset, epochs=5, steps_per_epoch=steps_per_epoch)


# In[ ]:




