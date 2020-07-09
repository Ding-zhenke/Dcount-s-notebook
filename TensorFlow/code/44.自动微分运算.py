#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow  as tf


# In[2]:


tf.__version__


# In[3]:


v = tf.Variable(0.0)


# In[4]:


(v + 1).numpy()


# In[5]:


v.assign(5)


# In[6]:


v


# In[7]:


v.assign_add(1)


# In[8]:


v


# In[9]:


v.read_value()


# In[10]:


w = tf.Variable([[1.0]])
with tf.GradientTape() as t:
    loss = w * w


# In[11]:


grad = t.gradient(loss, w)


# In[12]:


grad


# In[13]:


w = tf.constant(3.0)
with tf.GradientTape() as t:
    t.watch(w)
    loss = w * w


# In[14]:


dloss_dw = t.gradient(loss, w)


# In[15]:


dloss_dw


# In[16]:


w = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(w)
    y = w * w
    z = y * y


# In[17]:


dy_dw = t.gradient(y, w)


# In[18]:


dy_dw


# In[19]:


dz_dw = t.gradient(z, w)


# In[20]:


dz_dw


# In[64]:


(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()


# In[22]:


train_image.shape


# In[23]:


train_image = tf.expand_dims(train_image, -1)


# In[65]:


test_image = tf.expand_dims(test_image, -1)


# In[24]:


train_image.shape


# In[25]:


train_image = tf.cast(train_image/255, tf.float32)


# In[66]:


test_image = tf.cast(test_image/255, tf.float32)


# In[26]:


train_labels = tf.cast(train_labels, tf.int64)


# In[67]:


test_labels = tf.cast(test_labels, tf.int64)


# In[27]:


dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))


# In[68]:


test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))


# In[28]:


dataset


# In[29]:


dataset = dataset.shuffle(10000).batch(32)


# In[69]:


test_dataset = test_dataset.batch(32)


# In[30]:


dataset


# In[31]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3,3], activation='relu', input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(32, [3,3], activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10)
])


# In[32]:


optimizer = tf.keras.optimizers.Adam()


# In[33]:


loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[34]:


features, labels = next(iter(dataset))


# In[35]:


features.shape


# In[36]:


labels.shape


# In[37]:


predictions = model(features)


# In[38]:


predictions.shape


# In[39]:


tf.argmax(predictions, axis=1)


# In[40]:


labels


# In[41]:


def loss(model, x, y):
    y_ = model(x)
    return loss_func(y, y_)


# In[70]:


train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


# In[61]:


def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels, pred)


# In[71]:


def test_step(model, images, labels):
    pred = model(images)
    loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, pred)


# In[72]:


def train():
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels)
        print('Epoch{} loss is {}, accuracy is {}'.format(epoch,
                                                          train_loss.result(),
                                                          train_accuracy.result()))
        
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels)
        print('Epoch {} test_loss is {}, test_accuracy is {}'.format(epoch,
                                                              test_loss.result(),
                                                              test_accuracy.result()))
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


# In[73]:


train()


# ### tf.keras.metrics 汇总计算模块

# In[45]:


m = tf.keras.metrics.Mean('acc')


# In[46]:


m(10)


# In[47]:


m(20)


# In[49]:


m.result().numpy()


# In[50]:


m([30, 40])


# In[51]:


m.result()


# In[52]:


m.reset_states()


# In[54]:


m(1)


# In[55]:


m(2)


# In[56]:


m.result().numpy()


# In[57]:


a = tf.keras.metrics.SparseCategoricalAccuracy('acc')


# In[58]:


a(labels, model(features))


# In[59]:


1/32


# In[ ]:





# In[ ]:


#train()


# In[ ]:




