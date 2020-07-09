#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os


# In[2]:


print('Tensorflow Version: {}'.format(tf.__version__))


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(train_image, train_lable), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()


# In[5]:


train_image.shape


# In[6]:


train_lable.shape


# In[7]:


test_image.shape, test_label.shape


# In[8]:


plt.imshow(train_image[0])


# In[9]:


np.max(train_image[0])


# In[10]:


train_lable


# In[11]:


train_image = train_image/255
test_image = test_image/255


# In[12]:


train_image.shape


# In[13]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[14]:


model.summary()


# In[15]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[16]:


model.fit(train_image, train_lable, epochs=3)


# In[17]:


model.evaluate(test_image, test_label, verbose=0)


# # 保存整个模型

# 整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以为模型设置检查点，并稍后从完全相同的状态继续训练，而无需访问原始代码。
# 
# 在 Keras 中保存完全可正常使用的模型非常有用，您可以在 TensorFlow.js 中加载它们，然后在网络浏览器中训练和运行它们。
# 
# Keras 使用 HDF5 标准提供基本的保存格式。

# In[18]:


model.save('less_model.h5')


# In[19]:


new_model = tf.keras.models.load_model('less_model.h5')


# In[20]:


new_model.summary()


# In[21]:


new_model.evaluate(test_image, test_label, verbose=0)


# 此方法保存以下所有内容：
# 
# 1.权重值
# 2.模型配置（架构）
# 3.优化器配置

# # 仅保存架构

# 有时我们只对模型的架构感兴趣，而无需保存权重值或优化器。在这种情况下，可以仅保存模型的“配置” 。

# In[25]:


json_config = model.to_json()


# In[26]:


json_config


# In[27]:


reinitialized_model = tf.keras.models.model_from_json(json_config)


# In[28]:


reinitialized_model.summary()


# In[30]:


reinitialized_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['acc']
)


# In[31]:


reinitialized_model.evaluate(test_image, test_label, verbose=0)


# In[ ]:





# # 仅保存权重

# 有时我们只需要保存模型的状态（其权重值），而对模型架构不感兴趣。在这种情况下，可以通过get_weights()获取权重值，并通过set_weights()设置权重值

# In[32]:


weighs = model.get_weights()


# In[34]:


reinitialized_model.set_weights(weighs)


# In[35]:


reinitialized_model.evaluate(test_image, test_label, verbose=0)


# In[36]:


model.save_weights('less_weights.h5')


# In[37]:


reinitialized_model.load_weights('less_weights.h5')


# In[38]:


reinitialized_model.evaluate(test_image, test_label, verbose=0)


# # 在训练期间保存检查点

# 在训练期间或训练结束时自动保存检查点。这样一来，您便可以使用经过训练的模型，而无需重新训练该模型，或从上次暂停的地方继续训练，以防训练过程中断。

# 回调函数：tf.keras.callbacks.ModelCheckpoint

# In[18]:


checkpoint_path = 'training_cp/cp.ckpt'


# In[19]:


cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True)


# In[14]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[15]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[16]:


model.evaluate(test_image, test_label, verbose=0)


# In[20]:


model.load_weights(checkpoint_path)


# In[21]:


model.evaluate(test_image, test_label, verbose=0)


# In[ ]:





# In[ ]:





# In[18]:


model.fit(train_image, train_lable, epochs=3, callbacks=[cp_callback])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 自定义训练中保存检查点

# In[13]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10))


# In[14]:


optimizer = tf.keras.optimizers.Adam()


# In[15]:


loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[16]:


def loss(model, x, y):
    y_ = model(x)
    return loss_func(y, y_)


# In[17]:


def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels, pred)


# In[18]:


train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


# In[24]:


cp_dir = './customtrain_cp'
cp_prefix = os.path.join(cp_dir, 'ckpt')


# In[25]:


checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model
)


# In[ ]:





# In[20]:


dataset = tf.data.Dataset.from_tensor_slices((train_image, train_lable))


# In[21]:


dataset = dataset.shuffle(10000).batch(32)


# In[26]:


def train():
    for epoch in range(5):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels)
        print('Epoch{} loss is {}'.format(epoch, train_loss.result()))
        print('Epoch{} Accuracy is {}'.format(epoch, train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = cp_prefix)


# In[27]:


train()


# In[26]:


checkpoint.restore(tf.train.latest_checkpoint(cp_dir))


# In[29]:


tf.argmax(model(train_image, training=False), axis=-1).numpy()


# In[30]:


train_lable


# In[31]:


(tf.argmax(model(train_image, training=False), axis=-1).numpy() == train_lable).sum()/len(train_lable)


# In[ ]:





# In[ ]:





# 恢复模型

# In[ ]:





# In[ ]:





# In[ ]:




