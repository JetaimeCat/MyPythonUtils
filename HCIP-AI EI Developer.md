# HCIP-AI EI Developer

## 1. 神经网络基础

### 1.1. 深度学习预备知识

#### 1.1.1. 数据集

**特征（Feature）：**特征是用来描述机器学习系统处理的对象或事件的特性。

**样本（Sample）：**样本是指我们从某些希望机器学习系统处理的对象或事件中收集到的已经**量化**的特征的集合。

**数据集（Dataset）：**数据集是很多样本组成的集合。有时我们也将样本称为数据集中的数据点（Data Point）。

大部分机器学习算法可以被理解为在**数据集**上获取经验。

#### 1.1.2. 学习方法分类

**监督学习算法（Supervised Learning Algorithm）：**训练含有很多特征的数据集，不过数据集中的样本都有一个标签（Label）或目标（Target）。

**无监督学习算法（Unsupervised Learning Algorithm）：**训练含有很多特征的数据集，然后学习出这个数据集上有用的结构性质。

**监督**的理解：监督学习中，数据集提供标签给机器学习系统，通过标签与预测结果的对比进行学习。在无监督学习中，数据集并不提供标签，算法必须学会在没有正确答案的情况下理解、学习数据。

**半监督学习（Semi-Suppervised Learning）：**由于监督/非监督并非是严格分类的，很多机器学习技术会组合使用。半监督学习就是监督学习与无监督学习相结合的一种学习方法。半监督学习使用大量的未标记数据，以及同时使用标记数据作为学习的数据集。

**强化学习（Reinforcement Learning）：**并不使用一个固定的数据集用于训练模型。算法会和环境进行交互，所以学习系统和它的训练过程会有反馈回路。

#### 1.1.3. 凹凸函数

**凸集：**若集合中任意两点连线上的点都在该集合中，则称该集合为凸集。

**凸函数：**简单理解为在函数图像上任取两点$x_1,x_2$，如果函数图像以$x_1,x_2$作为端点绘制线段$l_1$，该线段$l_1$总在$[x_1,x_2]$函数图像的下方，该函数为凸函数。

**凹集：**若集合并非凸集，则该集合为凹集。

**凹函数：**简单理解为在函数图像上任取两点$x_1,x_2$，如果函数图像以$x_1,x_2$作为端点绘制线段$l_1$，该线段$l_1$总在$[x_1,x_2]$函数图像的上方，该函数为凹函数。

#### 1.1.4. 凸优化

**凸优化的定义：**

- 条件一：约束条件为凸集。
- 条件二：目标函数为凸函数。

**非凸优化问题转化为凸优化问题的方法：**

- 修改目标函数。
- 抛弃一些约束条件。

#### 1.1.5. 损失函数（代价函数，Loss Function）

损失函数衡量了评分函数的预测与真实样本标签的吻合度。

Loss的值都会设置为和吻合程度负相关。如果算法公式时正相关，定义损失函数时候加负号，调整为负相关。

#### 1.1.6. 交叉熵损失函

交叉熵损失（Cross Entropy / Softmax Loss）：对第$i$个样本$x_i$，神经网络$W$预测其为第$k$类的得分我们记作$f(x_{i},W)_{k}$，第$i$个样本$x_{i}$数据的真实标签类别是$y_{i}$。损失函数：
$$
\begin{align}
Loss_{i} &= -\sum_{k}p_{k}\log(\frac{e^{f}k}{\sum_{j}e^{f}j})\\
其中&p_{k}是x_{i}属于k类的概率,p_{k=y_{i}}=1,p_{k{\neq}y_{i}}=0.
\end{align}
$$

#### 1.1.7. 梯度下降

- 学习率（Learning Rate，LR），根据误差梯度调整权重数值的系数，通常记作$\eta$。通过学习率和梯度值更新所有参数值使得网络的损失函数值降低。

$$
\begin{align}
w^{+} &= w - \eta*\frac{\partial{Loss}}{\partial{w}}\\
\end{align}
$$

- 梯度下降常用的方法有三种

  - 批量梯度下降（BGD）：每次更新将使用所有的训练数据。但如果样本数据数目过多，更新速度将会很慢。
  - 随机梯度下降（SGD）：每次更新的时候只考虑了一个样本点，这样会大大加快训练速度，但是函数不一定是朝着极小值方向更新，且对噪点更加敏感。
  - 小批量梯度下降（MBGD）：MBGD解决了批量梯度下降法的训练速度慢的问题，以及随机梯度下降对噪声敏感的问题。

  

### 1.2. 人工神经网络

#### 1.2.1. 生物神经网络（Biological Neural Networks）





### 1.3. 深度前馈网络

### 1.4. 反向传播

### 1.5. 神经网络架构设计