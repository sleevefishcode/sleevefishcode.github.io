---
title: anaconda python环境搭建
date: 2022-03-11 20:05:35
categories: python
tags:  环境
---

##### anaconda python3.6环境下配置pytorch1.7 +cuda10.1

###### 从官网找到自己版本所对应的torch

https://pytorch.org/get-started/locally/

`conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1`

安装即可，速度过慢可用清华镜像源

 <!--more-->

###### 配置mmcv：

https://github.com/open-mmlab/mmcv#installation

参考：https://www.cnblogs.com/zhangly2020/p/14199156.html

注意对应的cuda和pytorch版本

> pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html![在这里插入图片描述](https://img-blog.csdnimg.cn/71ac0a0503bb42ad8fb8bd6c3f643362.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Zyf6LGG55qE54Ot54ix,size_19,color_FFFFFF,t_70,g_se,x_16)

参考：https://blog.csdn.net/qq_44442727/article/details/113444207

###### 安装项目所需包方法

###### ![requirement](requirement.png)

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

###### 配置mmdetection

参考：https://blog.csdn.net/lwuit/article/details/102430332

git clone https://github.com/open-mmlab/mmedetection.git /mmdetection（太慢timeout，换成了下面码云的）

注意版本要求，在码云上寻找自己需要的版本

> git clone https://gitee.com/hejuncheng1/mmdetection.git
>
> cd mmdetection
>
> pip install -e .

注意最后一个后面有.

同时用了python setup.py install  速度太慢timeout

pip --default-timeout=1000 install -e . 用这个会快很多

