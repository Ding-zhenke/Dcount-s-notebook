#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


tf.__version__


# In[3]:


tf.test.is_gpu_available()


# In[4]:


import os


# In[5]:


os.listdir('./annotations/trimaps')[-5:]


# In[6]:


img = tf.io.read_file('./annotations/trimaps/yorkshire_terrier_99.png')


# In[7]:


img = tf.image.decode_png(img)


# In[8]:


img.shape


# In[9]:


img = tf.squeeze(img)


# In[10]:


img.shape


# In[11]:


plt.imshow(img.numpy())
plt.show()


# In[12]:


img.numpy().max()


# In[13]:


img.numpy().min()


# In[14]:


np.unique(img.numpy())


# In[15]:


img = tf.io.read_file('./images/yorkshire_terrier_99.jpg')


# In[16]:


img = tf.image.decode_png(img)


# In[17]:


img.shape


# In[18]:


plt.imshow(img.numpy())
plt.show()


# In[19]:


import glob


# In[20]:


images = glob.glob('./images/*.jpg')


# In[21]:


images[:5]


# In[22]:


len(images)


# In[23]:


images.sort(key=lambda x: x.split('/')[-1])


# In[24]:


images[:5]


# In[25]:


images[-5:]


# In[26]:


annotations = glob.glob('./annotations/trimaps/*.png')


# In[27]:


annotations[:5]


# In[28]:


len(annotations)


# In[29]:


annotations.sort(key=lambda x: x.split('/')[-1])


# In[30]:


annotations[:5]


# In[31]:


annotations[-5:]


# In[32]:


len(images), len(annotations)


# In[33]:


np.random.seed(2019)
index = np.random.permutation(len(images))


# In[34]:


images = np.array(images)[index]


# In[35]:


images[:5]


# In[36]:


anno = np.array(annotations)[index]


# In[37]:


anno[:5]


# In[38]:


dataset = tf.data.Dataset.from_tensor_slices((images, anno))


# In[39]:


test_count = int(len(images)*0.2)


# In[40]:


test_count


# In[41]:


train_count = len(images) - test_count


# In[42]:


dataset_train = dataset.skip(test_count)


# In[43]:


dataset_test = dataset.take(test_count)


# In[44]:


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


# In[45]:


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# In[46]:


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/127.5 - 1
    input_mask -= 1
    return input_image, input_mask


# In[48]:


def load_image(input_image_path, input_mask_path):
    input_image = read_jpg(input_image_path)
    input_mask = read_png(input_mask_path)
    input_image = tf.image.resize(input_image, (224, 224))
    input_mask = tf.image.resize(input_mask, (224, 224))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# In[49]:


BATCH_SIZE = 8
BUFFER_SIZE = 100
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = test_count // BATCH_SIZE


# In[50]:


train = dataset_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset_test.map(load_image)


# In[51]:


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# In[52]:


train_dataset


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


for img, musk in train_dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(musk[0]))


# In[55]:


covn_base = tf.keras.applications.VGG16(weights='imagenet', 
                                        input_shape=(224, 224, 3),
                                        include_top=False)


# In[56]:


covn_base.summary()


# In[57]:


layer_names = [
    'block5_conv3',   # 14x14
    'block4_conv3',   # 28x28
    'block3_conv3',   # 56x56
    'block5_pool',
]
layers = [covn_base.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=covn_base.input, outputs=layers)

down_stack.trainable = False


# In[58]:


inputs = tf.keras.layers.Input(shape=(224, 224, 3))
o1, o2, o3, x = down_stack(inputs)
x1 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', 
                                     strides=2, activation='relu')(x)  # 14*14
x1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x1)  # 14*14
c1 = tf.add(o1, x1)    # 14*14
x2 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', 
                                     strides=2, activation='relu')(c1)  # 14*14
x2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x2)  # 14*14
c2 = tf.add(o2, x2)
x3 = tf.keras.layers.Conv2DTranspose(256, 3, padding='same', 
                                     strides=2, activation='relu')(c2)  # 14*14
x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)  # 14*14
c3 = tf.add(o3, x3)

x4 = tf.keras.layers.Conv2DTranspose(128, 3, padding='same', 
                                     strides=2, activation='relu')(c3)  # 14*14
x4 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x4)  # 14*14

predictions = tf.keras.layers.Conv2DTranspose(3, 3, padding='same', 
                                     strides=2, activation='softmax')(x4)

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)


# In[59]:


model.summary()


# In[60]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[61]:


EPOCHS = 20


# In[62]:


history = model.fit(train_dataset, 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)


# In[63]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


# In[64]:


num = 3


# In[65]:


for image, mask in test_dataset.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))


# In[66]:


for image, mask in train_dataset.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))


# In[67]:


model.save('fcn.h5')


# In[ ]:




