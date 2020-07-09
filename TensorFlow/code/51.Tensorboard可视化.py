#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow  as tf
import datetime


# In[2]:


tf.__version__


# In[3]:


(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()


# In[4]:


train_image.shape


# In[5]:


train_image = tf.expand_dims(train_image, -1)


# In[6]:


test_image = tf.expand_dims(test_image, -1)


# In[7]:


train_image.shape


# In[8]:


train_image = tf.cast(train_image/255, tf.float32)


# In[9]:


test_image = tf.cast(test_image/255, tf.float32)


# In[10]:


train_labels = tf.cast(train_labels, tf.int64)


# In[11]:


test_labels = tf.cast(test_labels, tf.int64)


# In[12]:


dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))


# In[13]:


test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))


# In[14]:


dataset


# In[15]:


dataset = dataset.repeat(1).shuffle(60000).batch(128)


# In[16]:


test_dataset = test_dataset.repeat(1).batch(128)


# In[17]:


dataset


# In[18]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3,3], activation='relu', input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(32, [3,3], activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[19]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


import datetime
import os


# In[21]:


log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[39]:


file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()


# In[41]:


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.02
    if epoch > 10:
        learning_rate = 0.01
    if epoch > 20:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


# In[22]:


model.fit(dataset,
          epochs=5,
          steps_per_epoch=60000//128,
          validation_data=test_dataset,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback])


# In[33]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# SCALARS 面板主要用于记录诸如准确率、损失和学习率等单个值的变化趋势。在代码中用 tf.summary.scalar() 来将其记录到文件中

# 每个图的右下角都有 3 个小图标，第一个是查看大图，第二个是是否对 y 轴对数化，第三个是如果你拖动或者缩放了坐标轴，再重新回到原始位置。

# GRAPHS 面板展示出你所构建的网络整体结构，显示数据流的方向和大小，也可以显示训练时每个节点的用时、耗费的内存大小以及参数多少。默认显示的图分为两部分：主图（Main Graph）和辅助节点（Auxiliary Nodes）。其中主图显示的就是网络结构，辅助节点则显示的是初始化、训练、保存等节点。我们可以双击某个节点或者点击节点右上角的 + 来展开查看里面的情况，也可以对齐进行缩放

# DISTRIBUTIONS 主要用来展示网络中各参数随训练步数的增加的变化情况，可以说是 多分位数折线图 的堆叠。

# HISTOGRAMS 和 DISTRIBUTIONS 是对同一数据不同方式的展现。与 DISTRIBUTIONS 不同的是，HISTOGRAMS 可以说是 频数分布直方图 的堆叠。

# # 记录自定义标量

# 重新调整回归模型并记录自定义学习率。这是如何做：
# 
# 使用创建文件编写器tf.summary.create_file_writer()。
# 
# 定义自定义学习率功能。这将被传递给Keras LearningRateScheduler回调。
# 
# 在学习率功能内，用于tf.summary.scalar()记录自定义学习率。
# 
# 将LearningRateScheduler回调传递给Model.fit（）。
# 
# 通常，要记录自定义标量，您需要使用tf.summary.scalar()文件编写器。文件编写器负责将此运行的数据写入指定的目录，并在使用时隐式使用tf.summary.scalar()。

# In[49]:


model.fit(dataset,
          epochs=30,
          steps_per_epoch=60000//128,
          validation_data=test_dataset,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback, lr_callback])


# In[ ]:





# In[ ]:





# # 自定义训练中使用Tensorboard

# In[19]:


optimizer = tf.keras.optimizers.Adam()


# In[20]:


loss_func = tf.keras.losses.SparseCategoricalCrossentropy()


# In[21]:


def loss(model, x, y):
    y_ = model(x)
    return loss_func(y, y_)


# In[22]:


train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


# In[23]:


def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels, pred)


# In[24]:


def test_step(model, images, labels):
    pred = model(images)
    loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, pred)


# In[25]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# In[26]:


def train():
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels)
            print('.', end='')
    
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels)
            print('*', end='')
            
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                               train_loss.result(), 
                               train_accuracy.result()*100,
                               test_loss.result(), 
                               test_accuracy.result()*100))
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


# In[27]:


train()


# In[28]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/gradient_tape')


# In[ ]:




