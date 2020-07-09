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
import numpy as np
import pathlib


# In[4]:


import pathlib


# In[5]:


data_dir = './dataset/2_class'


# In[6]:


data_root = pathlib.Path(data_dir)


# In[7]:


data_root


# In[8]:


for item in data_root.iterdir():
    print(item)


# In[9]:


all_image_paths = list(data_root.glob('*/*'))


# In[10]:


image_count = len(all_image_paths)


# In[11]:


all_image_paths[:3]


# In[12]:


all_image_paths[-3:]


# In[13]:


import random
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count


# In[14]:


all_image_paths[:5]


# In[15]:


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_names


# In[16]:


label_to_index = dict((name, index) for index,name in enumerate(label_names))
label_to_index


# In[17]:


all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


# In[18]:


all_image_labels[:5]


# In[19]:


import IPython.display as display


# In[20]:


def caption_image(label):
    return {0: 'airplane', 1: 'lake'}.get(label)


# In[21]:


for n in range(3):
    image_index = random.choice(range(len(all_image_paths)))
    display.display(display.Image(all_image_paths[image_index]))
    print(caption_image(all_image_labels[image_index]))
    print()


# 加载和格式化图像

# In[22]:


img_path = all_image_paths[0]
img_path


# In[23]:


img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")


# In[24]:


img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)


# In[25]:


img_tensor = tf.cast(img_tensor, tf.float32)
img_final = img_tensor/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


# In[26]:


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    return image


# In[27]:


import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(label))
print()


# In[28]:


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)


# In[29]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


# In[30]:


label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


# In[31]:


for label in label_ds.take(10):
    print(label_names[label.numpy()])


# In[32]:


image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# In[33]:


image_label_ds


# In[34]:


test_count = int(image_count*0.2)
train_count = image_count - test_count


# In[35]:


train_data = image_label_ds.skip(test_count)

test_data = image_label_ds.take(test_count)


# In[36]:


BATCH_SIZE = 32


# In[37]:


train_data = train_data.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=train_count))
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
train_data


# In[38]:


test_data = test_data.batch(BATCH_SIZE)gfc


# 建立模型

# In[39]:


model = tf.keras.Sequential()   #顺序模型
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[40]:


model.summary()


# In[41]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[42]:


steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE


# In[43]:


history = model.fit(train_data, epochs=30, steps_per_epoch=steps_per_epoch, validation_data=test_data, validation_steps=validation_steps)


# In[44]:


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




