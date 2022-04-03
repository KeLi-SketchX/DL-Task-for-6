# 用于信工6班深度学习建议

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

## 草图识别竞赛
1.草图数据库包含25个类别，每个类别800张草图，一共20，000张草图。20个类别作为seen类别，5个类别作为unseen类别，每个类别650张草图作为训练集，50张作为验证集，100张作为测试集

2.草图有2中存储方式，png格式（分辨率256x256）,svg格式（包含笔画顺序信息等）。

3.目的是锻炼大家的实践能力，我们有限比较resnet50单模型的精度，loss不限。不用quick draw的pretrain模型

4.衡量准确率和泛化能力

奖励和细节待补充。

| 渣本榜| seen | unseen |
| -- | --| -- |
| -- | --| -- |
| -- | --| -- |

