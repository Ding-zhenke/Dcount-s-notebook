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


data_dir = './dataset/moc'


# In[5]:


data_root = pathlib.Path(data_dir)


# In[6]:


data_root


# In[7]:


for item in data_root.iterdir():
    print(item)


# In[8]:


all_image_paths = list(data_root.glob('*/*'))


# In[9]:


image_count = len(all_image_paths)
image_count


# In[10]:


all_image_paths[:3]


# In[11]:


all_image_paths[-3:]


# In[12]:


import random
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)


# In[13]:


all_image_paths[:5]


# In[14]:


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_names


# In[15]:


color_label_names = set(name.split('_')[0] for name in label_names)
color_label_names


# In[16]:


item_label_names = set(name.split('_')[1] for name in label_names)
item_label_names


# In[17]:


color_label_to_index = dict((name, index) for index,name in enumerate(color_label_names))
color_label_to_index


# In[18]:


item_label_to_index = dict((name, index) for index,name in enumerate(item_label_names))
item_label_to_index


# In[19]:


all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
all_image_labels[:5]


# In[20]:


color_labels = [color_label_to_index[label.split('_')[0]] for label in all_image_labels]


# In[21]:


color_labels[:5]


# In[22]:


item_labels = [item_label_to_index[label.split('_')[1]] for label in all_image_labels]


# In[23]:


item_labels[:10]


# In[24]:


import IPython.display as display


# def caption_image(label):
#     return {0: 'airplane', 1: 'lake'}.get(label)

# In[25]:


for n in range(3):
    image_index = random.choice(range(len(all_image_paths)))
    display.display(display.Image(all_image_paths[image_index], width=100, height=100))
    print(all_image_labels[image_index])
    print()


# 加载和格式化图像

# In[26]:


img_path = all_image_paths[0]
img_path


# In[27]:


img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")


# In[28]:


img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)


# In[29]:


img_tensor = tf.cast(img_tensor, tf.float32)
img_tensor = tf.image.resize(img_tensor, [224, 224])
img_final = img_tensor/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


# In[30]:


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    image = 2*image-1
    return image


# In[31]:


import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow((load_and_preprocess_image(img_path) + 1)/2)
plt.grid(False)
plt.xlabel(label)
print()


# In[32]:


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)


# In[33]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


# In[34]:


label_ds = tf.data.Dataset.from_tensor_slices((color_labels, item_labels))


# In[35]:


for ele in label_ds.take(3):
    print(ele[0].numpy(), ele[1].numpy())


# In[36]:


image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# In[37]:


image_label_ds


# In[38]:


test_count = int(image_count*0.2)
train_count = image_count - test_count


# In[39]:


train_data = image_label_ds.skip(test_count)

test_data = image_label_ds.take(test_count)


# In[40]:


BATCH_SIZE = 16


# In[41]:


train_data = train_data.shuffle(buffer_size=train_count).repeat(-1)
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
train_data


# In[42]:


test_data = test_data.batch(BATCH_SIZE)


# # 建立模型

# In[43]:


mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), 
                                               include_top=False,
                                               weights='imagenet')


# In[44]:


mobile_net.trianable = False


# In[45]:


inputs = tf.keras.Input(shape=(224, 224, 3))


# In[46]:


x = mobile_net(inputs)


# In[47]:


x.get_shape()


# In[48]:


x = tf.keras.layers.GlobalAveragePooling2D()(x)


# In[49]:


x.get_shape()


# In[50]:


x1 = tf.keras.layers.Dense(1024, activation='relu')(x)
out_color = tf.keras.layers.Dense(len(color_label_names), 
                                  activation='softmax',
                                  name='out_color')(x1)


# In[51]:


x2 = tf.keras.layers.Dense(1024, activation='relu')(x)
out_item = tf.keras.layers.Dense(len(item_label_names), 
                                 activation='softmax',
                                 name='out_item')(x2)


# In[52]:


model = tf.keras.Model(inputs=inputs,
                       outputs=[out_color, out_item])


# In[53]:


model.summary()


# In[54]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'out_color':'sparse_categorical_crossentropy',
                    'out_item':'sparse_categorical_crossentropy'},
              metrics=['acc']
)


# In[55]:


train_steps = train_count//BATCH_SIZE
test_steps = test_count//BATCH_SIZE


# In[56]:


model.fit(train_data,
          epochs=15,
          steps_per_epoch=train_steps,
          validation_data=test_data,
          validation_steps=test_steps
)


# In[57]:


model.evaluate(test_data)


# In[59]:


my_image = load_and_preprocess_image(r'D:\163\tf20\jk\dataset\moc\blue_jeans\00000004.jpg')


# In[62]:


my_image.shape


# In[63]:


my_image = tf.expand_dims(my_image, 0)


# In[64]:


my_image.shape


# In[66]:


pred = model.predict(my_image)


# In[69]:


np.argmax(pred[0])


# In[70]:


np.argmax(pred[1])


# In[ ]:




