# Simple-GAN-Base-on-Matlab
* simple generative adversarial networks base on matlab 
* 基于matlab上的简单生成对抗性网络
## mnist_uint8数据集准备
* [mnist_uint8](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat)
## 网络结构
* generator 
<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/5.png"/>
</p>

* generator + discriminator
<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/6.png"/>
</p>

## 代码

* gan_adam.m

推荐使用ADAM优化器对GAN网络进行更新

结果示例：

<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/1.png"/>
</p>

<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/2.png"/>
</p>

* gan_sgd.m

使用SGD算法对GAN网络进行迭代更新

结果示例：

<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/3.png"/>
</p>

<p align="center">
  <img src="https://github.com/JZhaoCH/Simple-GAN-Base-on-Matlab/blob/master/readme%20images/4.png"/>
</p>
