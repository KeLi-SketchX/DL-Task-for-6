# 用于信工6班深度学习建议
通过布置具体的任务来锻炼大家的深度学习动手能力，希望大家在大三时能够具备参加竞赛或者论文发表工作的能力。  

我会不定时的给大家集中答疑，平时也会在群里答疑。  

不定时邀请优秀的同学分享。

## 环境安装
1.安装conda https://docs.conda.io/en/latest/miniconda.html  
2.创建虚拟环境   
3.利用conda安装PyTorch torchvision opencv Pillow matplotlib等环境。 https://pytorch.org/  
pytorch有gpu版本和cpu版本，注意区分。同时pytorch，torchvision和cudatoolkit版本需要相互匹配，cudatoolkit版本和你的显卡驱动相互匹配。
nvidia-smi 查看gpu信息  

这个阶段需要大家掌握深度学习中常用的软件的安装。

## Mnist手写体识别 
1.Mnist数据库可以从[这里](http://yann.lecun.com/exdb/mnist/)下载，也可以在pytorch中torchvision的datasets.MNIST中直接访问。  

2.模型结构推荐大家用Lenet-5, Resnet18等小模型。注意图片的通道数目和分辨率是否和模型要求的相匹配。  

3.了解分类任务常用的评价指标，Accuracy，precision,recall, F1 score,CMC等，根据最终目的选择合适的评价指标和模型的损失函数。

4.撰写代码，了解深度学习代码的大概框架--数据loader,模型创建，优化器创建，损失函数创建，加载数据，模型前向推理并计算损失，后向传播更新模型，模型验证，日志存储等

5.代码能够运行完后，分析loss和准确率的变化，调整模型超参数（学习率，学习率变化方式，batchsize等等）。

这个阶段需要大家掌握：

1.图像分类任务常用的评价指标

2.卷积神经网络的基本构成

3.深度学习代码的基本构成和撰写

4.基本的调参数方式

窦越嘉同学贡献了他的[代码](https://github.com/Eason0921/Sanitater)，大家可以去交流


## 草图识别竞赛 （4.11-5.11）
1.草图数据库包含25个类别，每个类别800张草图，一共20，000张草图。20个类别作为seen类别，5个类别作为unseen类别，每个类别650张草图作为训练集，50张作为验证集，100张作为测试集

2.草图有2中存储方式，png格式（分辨率256x256）,svg格式（包含笔画顺序信息等）。

3.目的是锻炼大家的实践能力，我们有限比较resnet50单模型的精度，loss不限，辅助任务不限。不用quick draw的pretrain模型

4.衡量准确率和泛化能力
5.建议补充各种分类loss（center loss，focal loss， arcface loss等等）和其调优细节

数据，奖励，细节，评测代码待补充。

##SPG数据集分类
| 渣本榜| seen | unseen |简介（可选）｜
| -- | --| -- |--｜
| -- | --| -- |--｜
| -- | --| -- |--｜

##TUBerlin分类
| 渣本榜 | accuracy| 简介（可选）|
｜胡柒柒｜78.95｜baseline代码｜


