# 用于信工6班深度学习建议

## 环境安装
1.安装conda   
https://docs.conda.io/en/latest/miniconda.html  
2.创建虚拟环境   
3.利用conda安装PyTorch torchvision opencv Pillow matplotlib等环境  
https://pytorch.org/  
pytorch有gpu版本和cpu版本，注意区分。nvidia-smi 查看gpu信息  

## Mnist手写体识别
Mnist数据库可以从[这里](http://yann.lecun.com/exdb/mnist/)下载，也可以在pytorch中torchvision的datasets.MNIST中直接访问。
模型结构推荐大家用Lenet-5, Resnet18等小模型。注意图片的通道数目和分辨率是否和模型要求的相匹配。
