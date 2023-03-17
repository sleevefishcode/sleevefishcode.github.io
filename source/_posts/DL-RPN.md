---
title: DL-RPN
date: 2022-03-28 15:14:59
categories: python
tags: 深度学习
---

在学习faster R-CNN时遇到了RPN**区域推荐网络**，记录一下。
<!--more-->

在faster R-CNN引入了RPN后，它能够利用Backbone出来的卷积层featuremap学习出proposal cadidates，这个方法取代传统的离线方式，并且能够很方便的整合到任何目标检测网络中，实现e2e训练。

![img](DL-RPN/faster R-CNN)

> ​									faster R-CNN 结构图

RPN在结构中是用来生成一系列proposals的一层，本质上它是Backbone的最后一层卷积层作为输入，proposal的具体位置作为输出的神经网络。

RPN主要包含以下几步：

- 生成Anchor boxes.
- 判断anchor boxes包含的是前景还是背景.
- 回归学习anchor boxes和groud truth的标注的位置差，来精确定位物体

首先介绍下什么是anchor和anchor boxes。每个anchor点就是backbone网络最后一层卷积层feature map上的元素点。而anchor boxes是以anchor为中心点而生成的一系列框boxes。一个anchor对应的框的具体数量是由scales和aspect ratios 2个参数控制。scales指的是对于每种形状，框体最长边的像素大小，aspect ratios指的是具体由哪些形状，描述的是长宽比。所以scales[8,16,32] 和ratios [0.5,1,2]就代表一个anchor会生成9个anchor boxes。注意的是anchor boxes的坐标是对应在原图尺寸上的，而feature map相比原图是缩小很多倍的，比如VGG16,backbone网络出来的图片尺寸比原图缩小了16倍，如果在原图中采anchor boxes，就需要按16像素点跳着采样。

**RPN的核心概念------anchor**

有一些参考文献说Anchor 是大小和尺寸固定的候选框，这种说法有一定道理，但不准确。在回答什么是anchor之前，先看一下RPN网络的一个第一步运算，RPN的第一步运算实际上就是一个3*3*256的卷积运算，我们称3*3为一个滑动窗口（sliding window)，假设RPN的输入是13*13*256的特征图，然后使用3*3*256的卷积核进行卷积运算，最后依然会得到一个a*a*256的特征图，这里的a与卷积的步长有关。

    在原始论文中，作者选定了3种不同scale,3种不同宽高比（aspect ratios)的矩形框作为“基本候选框”，

三种scale/size是{128，256，512}

三种比例{1：1， 1：2， 2：1}

故而一共是3x3=9种，很明显，这9种基本候选框的长宽远远大于特征图的长宽，所以这9个指的应该是原始图像。




有了anchor boxes还不够，这些框体大多都都没有包含目标物体，所以我们需要模型学习判断box是否包含物体（即是前景图片还是背景图片），同时对于前景图片我们还要学习来预测box和GT的offset。这2个任务分别是用2个卷积网络实现的，如下图所示，

![img](DL-RPN/cls-reg layer)

​									*cls layer和reg layer*

假设每个anchor生成了k个boxes，每个anchor box会输入到2个卷积网络，分别是cls layaer和reg layer。

![img](DL-RPN/convolution)

cls layer是前景判断任务，最后会输出每个anchor box的前景分数和背景分数。以上图为例，假设anchor boxes数量是9，经过cls layer的1*1*18卷积核输出就是H*W*18的feature map，每个anchor的维度是18=2*9，就代表每个anchor作为前景和背景的分数。

而reg layer的输出是boxes与GT的（x,y,w,h）的offsets，（x,y）是box中心点的坐标。（w,h）是宽和高。以上图为例，经过reg layer的1*1*36的卷积核之后输出的是H*W*36的feature map，每个anchor的维度是36=4*9，也就代表了（w,y,w,h）的offsets。anchor boxes会根据模型预测出来的offset来调整尺寸，作为RPN的输出。

RPN的训练数据是通过比较anchor boxes和GT boxes生成的，会在图片中采样一些anchor boxes，然后计算anchor box和GT box的IOU来判断该box是前景还是背景，对于是前景的box还要计算其与GT box之间各个坐标offset。训练数据一般按照正负样例1：3的比例来训练模型。下图是最终的损失函数表达式，包含了前景判断和坐标回归，前者训练使用的损失函数是交叉熵损失函数cross-entropy loss，后者用的是smooth l1 loss。

![img](DL-RPN/loss)

上图前半段是前景判断loss，pi指的是前景的概率分数，*是GT的值。后半段loss是坐标的分类模型loss，t是各坐标的vector向量。



作用：在faster rcnn里面，anchor（或者说RPN网络）的作用是代替以往rcnn使用的selective search的方法寻找图片里面可能存在物体的区域。当一张图片输入resnet或者vgg，在最后一层的feature map上面，寻找可能出现物体的位置，这时候分别以这张feature map的每一个点为中心，在原图上画出9个尺寸不一anchor。然后计算anchor与GT（ground truth） box的iou（重叠率），满足一定iou条件的anchor，便认为是这个anchor包含了某个物体。

目标检测的思想是，首先在图片中寻找**“可能存在物体的位置（regions）”**，然后再判断**“这个位置里面的物体是什么东西”**，所以region proposal就参与了判断物体可能存在位置的过程。

**region proposal是让模型学会去看哪里有物体，GT box就是给它进行参考，告诉它是不是看错了，该往哪些地方看才对。**







参考：

链接：https://blog.csdn.net/weixin_38145317/article/details/90753704

链接：https://www.zhihu.com/question/265345106/answer/292998341

https://zhuanlan.zhihu.com/p/338217417

