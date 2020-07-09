#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from lxml import etree
from matplotlib.patches import Rectangle
import numpy as np
import glob

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
# In[2]:


tf.__version__


# In[3]:


tf.test.is_gpu_available()


# In[4]:


img = tf.io.read_file(r'./images/Abyssinian_1.jpg')


# In[5]:


img = tf.image.decode_jpeg(img)


# In[6]:


img.shape


# In[7]:


plt.imshow(img.numpy())
plt.show()


# In[8]:


xml = open(r'./annotations/xmls/Abyssinian_1.xml').read()


# In[9]:


sel = etree.HTML(xml)


# In[10]:


width = sel.xpath('//size/width/text()')[0]
width


# In[11]:


height = sel.xpath('//size/height/text()')[0]
height


# In[12]:


xmin = sel.xpath('//bndbox/xmin/text()')[0]
xmin


# In[13]:


ymin = sel.xpath('//bndbox/ymin/text()')[0]
xmax = sel.xpath('//bndbox/xmax/text()')[0]
ymax = sel.xpath('//bndbox/ymax/text()')[0]


# In[14]:


ymin, xmax, ymax


# In[15]:


width = int(width)
height = int(height)


# In[16]:


xmin = int(xmin)
ymin = int(ymin)
xmax = int(xmax)
ymax = int(ymax)


# In[17]:


plt.imshow(img.numpy())
rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
ax = plt.gca()
ax.axes.add_patch(rect)
plt.show()


# In[18]:


img = tf.image.resize(img, [224, 224])


# In[19]:


img = img/255


# In[20]:


xmin = xmin*224/width
ymin = ymin*224/height
xmax = xmax*224/width
ymax = ymax*224/height


# In[21]:


plt.imshow(img.numpy())
rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
ax = plt.gca()
ax.axes.add_patch(rect)
plt.show()


# In[ ]:





# # 创建输入管道

# In[22]:


images = glob.glob('./images/*.jpg')


# In[23]:


images[:5]


# In[24]:


len(images)


# In[25]:


xmls = glob.glob('./annotations/xmls/*.xml')


# In[26]:


xmls[:5]


# In[27]:


images[-5:]


# In[28]:


len(xmls)


# In[29]:


xmls_names = [x.split('\\')[-1].split('.xml')[0] for x in xmls]


# In[30]:


xmls_names[:3]


# In[31]:


len(xmls_names)


# In[32]:


images_names = [x.split('\\')[-1].split('.jpg')[0] for x in images]


# In[33]:


images_names[:3]


# In[34]:


names = list(set(images_names)&set(xmls_names))


# In[35]:


names[:3]


# In[36]:


len(names)


# In[37]:


imgs = [img for img in images if img.split('\\')[-1].split('.jpg')[0] in names]


# In[38]:


len(imgs)


# In[39]:


imgs[:3]


# In[40]:


imgs[-3:]


# In[41]:


imgs.sort(key=lambda x: x.split('\\')[-1].split('.jpg')[0])


# In[42]:


imgs[:3]


# In[43]:


imgs[-3:]


# In[44]:


xmls.sort(key=lambda x: x.split('\\')[-1].split('.xml')[0])


# In[45]:


xmls[:3]


# In[46]:


xmls[-3:]


# In[47]:


scal = 224


# In[48]:


def to_labels(path):
    xml = open(r'{}'.format(path)).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    height = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//bndbox/xmin/text()')[0])
    ymin = int(sel.xpath('//bndbox/ymin/text()')[0])
    xmax = int(sel.xpath('//bndbox/xmax/text()')[0])
    ymax = int(sel.xpath('//bndbox/ymax/text()')[0])
    return [xmin/width, ymin/height, xmax/width, ymax/height]


# In[49]:


labels = [to_labels(path) for path in xmls]


# In[50]:


out1_label, out2_label, out3_label, out4_label = list(zip(*labels))


# In[51]:


len(out1_label), len(out2_label), len(out3_label), len(out4_label)


# In[52]:


index = np.random.permutation(len(imgs))


# In[53]:


images = np.array(imgs)[index]
#labels = labels[index]


# In[54]:


out1_label = np.array(out1_label)[index]
out2_label = np.array(out2_label)[index]
out3_label = np.array(out3_label)[index]
out4_label = np.array(out4_label)[index]


# In[55]:


label_datset = tf.data.Dataset.from_tensor_slices((
                                              out1_label, 
                                              out2_label, 
                                              out3_label, 
                                              out4_label))


# In[56]:


image_dataset = tf.data.Dataset.from_tensor_slices(images)


# In[57]:


label_datset


# In[58]:


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


# In[59]:


def normalize(input_image):
    input_image = tf.image.resize(input_image, [scal, scal])
    input_image = tf.cast(input_image, tf.float32)/127.5 - 1
    return input_image


# In[60]:


@tf.function
def load_image(input_image_path):
    input_image = read_jpg(input_image_path)
    input_image = normalize(input_image)
    return input_image


# In[61]:


image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# In[62]:


image_dataset


# In[63]:


dataset = tf.data.Dataset.zip((image_dataset, label_datset))


# In[64]:


dataset


# In[65]:


test_count = int(len(images)*0.2)


# In[66]:


test_count


# In[67]:


train_count = len(images) - test_count


# In[68]:


dataset_train = dataset.skip(test_count)


# In[69]:


dataset_test = dataset.take(test_count)


# In[70]:


BATCH_SIZE = 8
BUFFER_SIZE = 300
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = test_count // BATCH_SIZE


# In[71]:


train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = dataset_test.batch(BATCH_SIZE)


# In[72]:


train_dataset


# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


for img, label in train_dataset.take(1):
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    out1, out2, out3, out4 = label
    xmin, ymin, xmax, ymax = (out1[0]).numpy()*scal, (out2[0]).numpy()*scal, (out3[0]).numpy()*scal, (out4[0]).numpy()*scal
    rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)


# In[ ]:





# # 创建定位模型

# In[76]:


xception = tf.keras.applications.Xception(weights='imagenet', 
                                          include_top=False,
                                          input_shape=(scal, scal, 3))


# In[77]:


xception.trianable = False


# In[78]:


inputs = tf.keras.layers.Input(shape=(scal, scal, 3))

x = xception(inputs)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)

out1 = tf.keras.layers.Dense(1)(x)
out2 = tf.keras.layers.Dense(1)(x)
out3 = tf.keras.layers.Dense(1)(x)
out4 = tf.keras.layers.Dense(1)(x)

predictions = [out1, out2, out3, out4]

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)


# In[79]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='mse',
              metrics=['mae'])


# In[80]:


EPOCHS = 50


# In[81]:


history = model.fit(train_dataset, 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)


# In[82]:


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


# # 模型保存

# In[84]:


model.save('detect_v1.h5')


# In[75]:


new_model = tf.keras.models.load_model('detect_v1.h5')


# In[77]:


plt.figure(figsize=(8, 24))
for img, _ in test_dataset.take(1):
    out1, out2, out3, out4 = new_model.predict(img)
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
        xmin, ymin, xmax, ymax = out1[i]*224, out2[i]*224, out3[i]*224, out4[i]*224,
        rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
        ax = plt.gca()
        ax.axes.add_patch(rect)


# In[ ]:




