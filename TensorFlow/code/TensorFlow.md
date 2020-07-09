<font face = "楷体">


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [一、简介与理论基础](#一简介与理论基础)
  - [1. 简介](#1-简介)
  - [2. 学习内容](#2-学习内容)
  - [3. 安装](#3-安装)
- [二、机器学习基础与keras基础](#二机器学习基础与keras基础)
  - [1. Keras基础](#1-keras基础)
    - [1.1 库文件引入](#11-库文件引入)
    - [1.2 版本检查](#12-版本检查)
    - [1.3 观察神经网络结构](#13-观察神经网络结构)
  - [2. 线性回归](#2-线性回归)
  - [3. 多层感知器](#3-多层感知器)
  - [4. 机器学习基础](#4-机器学习基础)
  - [5. 问题分类](#5-问题分类)
    - [5.1 概率问题——逻辑回归问题](#51-概率问题逻辑回归问题)
    - [5.2 多分类问题](#52-多分类问题)
  - [6. 模型优化](#6-模型优化)
    - [6.1 过拟合](#61-过拟合)
    - [6.2 欠拟合](#62-欠拟合)
    - [6.3 提高拟合能力](#63-提高拟合能力)
- [三、tf.data模块](#三tfdata模块)

<!-- /code_chunk_output -->

---

# 一、简介与理论基础

---

## 1. 简介

- 通过清理废弃的API和减少重复来简化 API。
- 在训练方面：
  - 使用 Keras 和 eager execution 轻松构建模型，为研究提供强大的实验工具。
- Tf.keras：允许创建复杂的拓扑，包括使用残差层、自定义多输入/输出模型以及强制编写的正向传递。轻松创建自定义训练循环。低级 TensorFlow API 始终可用，并与更高级别的抽象一起工作，以实现完全可定制的逻辑。
- 在任意平台上实现稳健的生产环境模型部署不论是在服务器、边缘设备还是网页上，也不论你使用的是什么语言或平台，TensorFlow 总能让你轻易训练和部署模型。
- 通过标准化交换格式来改进跨平台和跨语言部署
  
## 2. 学习内容

- tf.keras构建和训练模型的核心高级 API
- 单输入单输出Sequential 顺序模型
- 函数式API
- Eager模式与自定义训练，直接迭代和直观调试，Eager模式下求解梯度与自定义训练
  - Eager模式：直接迭代和直观调试
  - tf.GradientTape求解梯度，自定义训练逻辑
- tf.data：加载图片数据与结构化数据
- 介绍 tf.fuction：自动图运算
- CNN
- 多输出卷积神经网络综合实例
- 迁移学习
- 模型保存与可视化

## 3. 安装

- 安装 GPU，NVIDIA 算力要大于3.0：<https://developer.nvidia.com/cuda-gpus>
- 安装 NVIDIA相关软件必须包括：<https://developer.nvidia.com/cuda-gpus>
  - （一） NVIDIA驱动程序
  - （二） CUDA
  - （三） cudnn

---

# 二、机器学习基础与keras基础

---

## 1. Keras基础

### 1.1 库文件引入

以下是可能会用到的模块，需先安装好
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

### 1.2 版本检查

使用`print(tf.__version__)`即可

### 1.3 观察神经网络结构

- 观察样式`model.summary()`查看结构，看看元素和内部
- 申请预测`model.predict(x)`或`model.predict(pd.Series([20]))`#预测几个

## 2. 线性回归

首先读入数据文件，建立一层的神经元即可拟合

```python
data = pd.read_csv('./dataset/Income1.csv')
x = data.Education
y = data.Income
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()
model.compile(optimizer='adam',
              loss='mse'
)
history = model.fit(x, y, epochs=5000)
model.predict(x)
model.predict(pd.Series([20]))
```

加上激活函数后可以拟合二次方程，甚至是`sin`

## 3. 多层感知器

读入数据后设两层神经元

```python
import tensorflow as tf
mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('dataset/Advertising.csv')
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)]
)
model.summary()
model.compile(optimizer='adam',
              loss='mse'
)
model.fit(x, y, epochs=100)
test = data.iloc[:10, 1:-1]
model.predict(test)
test = data.iloc[:10, -1]
```

## 4. 机器学习基础

线性回归预测的是一个连续值，逻辑回归给出的”是”和“否”的回答

- 逻辑回归损失函数
    >平方差所惩罚的是与损失为同一数量级的情形，对于分类问题，我们最好的使用交叉熵损失函数会更有效，交叉熵会输出一个更大的“损失”
- 交叉熵损失函数
    >交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。假设概率分布p为期望输出，概率分布q为实际输出，H(p,q)为交叉熵，则：>$$
    >H(p,q)=-\sum_{x}\log_{10}{q(x)} 
    >$$
- softmax分类
    >对数几率回归解决的是二分类的问题，对于多个选项的问题，我们可以使用softmax函数,它是对数几率回归在 N 个可能不同的值上的推广神经网络的原始输出不是一个概率值，实质上只是输入的数值做了复杂的加权和与非线性处理之后的一个值而已，那么如何将这个输出变为概率分布？这就是Softmax层的作用。
    >softmax要求每个样本必须属于某个类别，且所有可能的样本均被覆盖。
    >softmax个样本分量之和为1，只有两个样本时，与对数几率回归完全相同
    >$$
    >\sigma (z)_{j} =\frac{e^{z_j} }{\sum_{k=1}^{K}e^{z_j}  } ,for j = 1,...,K
    >$$
- 网络容量：可以认为与网络中的可训练参数成正比
  >网络中的神经单元数越多，层数越多，神经网络的拟合能力越强。但是训练速度、难度越大，越容易产生过拟合。
- 设定优化器和损失函数
```python
model.compile(optimizer='adma',loss='mse')
tf.keras.optimizers.Adam(lr=0.01)#可以这样做，但是一般默认
history = model.fit(x,y,epochs=1000)#训练参数,训练次数
history.history.keys()记录了[loss,acc]准确率
plt.plot(history.epoch,history.loss)  画一下训练轮数和损失
```

## 5. 问题分类

### 5.1 概率问题——逻辑回归问题

非黑即白
>逻辑回归，一般用sigmod函数，其值域类似概率值（0，1），这个是分类问题，即二元分类，是or否，最后一个要用sigmoid,前面用relu
>损失函数最好用交叉熵损失函数，求的是与概率分布之间的损失`binary_cossentropy`。二元交叉熵计算正确率`metircs=['acc']`

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#读取数据
data = pd.read_csv('dataset/credit-a.csv', header=None)
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)
#建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x, y, epochs=100)
plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))
```

### 5.2 多分类问题

- `categorical_crossentropy`用独热编码[0，1，0，1，1，0]
- `sparse_categorical_crossentropy`，不是独,其实就是顺序编码，比如用2表示鞋子，38表示衣服

```python
#输出十个概率值，用softmax激活把多个输出变成概率分布
model.add(tf.keras.layers.Dense(10,activation='softmax'))
#Opitimizer 注意选交叉熵
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

tf.kearas.utils.to_categorical(编码)#转成独热编码
```
```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#读取数据
(train_image, train_lable),
(test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = train_image/255
test_image = test_image/255
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)
model.fit(train_image, train_lable, epochs=5)
model.evaluate(test_image, test_label)
```

## 6. 模型优化

### 6.1 过拟合

>测试数据集分低，训练数据集分高

### 6.2 欠拟合

>两个的分低

### 6.3 提高拟合能力

- 增加层，效果好
- 增加隐藏神经元个数，效果一般
参数选择原则：首先开发一个过拟合的模型：
- (1) 添加更多的层。
- (2) 让每一层变得更大。
- (3) 训练更多的轮次
抑制过拟合：

- （1）dropout
    >取平均的作用,减少神经元之间复杂的共适应关系,Dropout类似于性别在生物进化中的角色
    >`keras.layers.Dropout(0.5)`每一层之后加上即可
- （2）正则化
- （3）图像增强

---

# 三、tf.data模块

---
基于 `tf.data` API，我们可以使用简单的代码来构建复杂的输入，`tf.data` API 可以轻松处理大量数据、不同的数据格式以及复杂的转换。
`tf.data.Dataset`:表示一系列元素,每个元素包含一个或多个 Tensor 对象。例如，在图片管道中，一个元素可能是单个训练样本，具有一对表示图片数据和标签的张量。
