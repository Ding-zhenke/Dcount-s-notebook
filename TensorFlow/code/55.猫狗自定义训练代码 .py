#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import glob
import os


# In[ ]:


print('Tensorflow version: {}'.format(tf.__version__))


# In[ ]:


train_image_path = glob.glob('../input/training_set/training_set/*/*.jpg')


# In[ ]:


np.random.shuffle(train_image_path)


# In[ ]:


len(train_image_path)


# In[ ]:


train_image_path[:5]


# In[ ]:


train_image_path[-5:]


# In[ ]:


train_image_label = [int(path.split('training_set/training_set/')[1].split('/')[0] == 'cats') for path in train_image_path]


# In[ ]:


train_image_label[-5:]


# In[ ]:


def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
#    image = tf.image.random_crop(image, [256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
#    image = tf.image.random_brightness(image, 0.5)
#    image = tf.image.random_contrast(image, 0, 1)
    image = tf.cast(image, tf.float32)
    image = image/255
    label = tf.reshape(label, [1])
    return image, label


# In[ ]:


#[1, 2, 3]  -->  [[1], [2], [3]]


# In[ ]:


#tf.image.convert_image_dtype


# In[ ]:


train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[ ]:


train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)


# In[ ]:


train_image_ds


# In[ ]:


for img, label in train_image_ds.take(1):
    plt.imshow(img)


# In[ ]:


BATCH_SIZE = 32
train_count = len(train_image_path)


# In[ ]:


train_image_ds = train_image_ds.shuffle(500).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE)


# In[ ]:


test_image_path = glob.glob('../input/test_set/test_set/*/*.jpg')
test_image_label = [int(path.split('test_set/test_set/')[1].split('/')[0] == 'cats') for path in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.prefetch(AUTOTUNE)


# In[ ]:


len(test_image_path)


# In[ ]:


model = keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])


# In[ ]:


model.summary()


# In[ ]:


tf.keras.losses.binary_crossentropy([0.,0.,1.,1.], [1.,1.,1.,1.])


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


# In[ ]:


epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy()

epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy()


# In[ ]:


train_accuracy([1,0,1], [1,1,1])


# In[ ]:


def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuracy(labels, tf.cast(pred>0, tf.int32))


# In[ ]:


def test_step(model, images, labels):
    pred = model(images, training=False)
    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    epoch_loss_avg_test(loss_step)
    test_accuracy(labels, tf.cast(pred>0, tf.int32))


# In[ ]:


train_loss_results = []
train_acc_results = []

test_loss_results = []
test_acc_results = []


# In[ ]:


num_epochs = 30


# In[ ]:


for epoch in range(num_epochs):
    for imgs_, labels_ in train_image_ds:
        train_step(model, imgs_, labels_)
        print('.', end='')
    print()
    
    train_loss_results.append(epoch_loss_avg.result())
    train_acc_results.append(train_accuracy.result())
    
    
    for imgs_, labels_ in test_image_ds:
        test_step(model, imgs_, labels_)
        
    test_loss_results.append(epoch_loss_avg_test.result())
    test_acc_results.append(test_accuracy.result())
    
    print('Epoch:{}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}'.format(
        epoch + 1,
        epoch_loss_avg.result(),
        train_accuracy.result(),
        epoch_loss_avg_test.result(),
        test_accuracy.result()
    ))
    
    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()
    
    epoch_loss_avg_test.reset_states()
    test_accuracy.reset_states()


# In[ ]:





# In[ ]:




