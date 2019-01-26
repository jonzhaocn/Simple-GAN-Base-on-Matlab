# Simple-GAN-Base-on-Matlab
* simple generative adversarial networks base on matlab 
* 基于matlab上的简单生成对抗性网络
## mnist_uint8数据集准备
* [mnist_uint8](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat)
## 网络结构
* generator 
<p align="center">
    <a href="/readme-images/generator.png">
  		<img src="/readme-images/generator.png"/>
    </a>
</p>


* generator + discriminator
<p align="center">
    <a href="/readme-images/generator-discriminator.png">
  		<img src="/readme-images/generator-discriminator.png"/>
    </a>
</p>


## 代码

* gan_adam.m

推荐使用ADAM优化器对GAN网络进行更新

结果示例：

<p align="center">
    <a href="/readme-images/adam-result-1.png">
  		<img src="/readme-images/adam-result-1.png"/>
    </a>
</p>

<p align="center">
    <a href="/readme-images/adam-result-2.png">
  		<img src="/readme-images/adam-result-2.png"/>
    </a>
</p>


* gan_sgd.m

使用SGD算法对GAN网络进行迭代更新

结果示例：

<p align="center">
    <a href="/readme-images/sgd-result-1.png">
  		<img src="/readme-images/sgd-result-1.png"/>
    </a>
</p>

<p align="center">
    <a href="/readme-images/sgd-result-2.png">
  		<img src="/readme-images/sgd-result-2.png"/>
    </a>
</p>

