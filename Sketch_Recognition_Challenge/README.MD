# 草图识别挑战-SPG

1、数据来源于SPG-dataset中的草图，是Quick Draw的一个子集，因此为了公平，不使用任何和Quick Draw相关的数据和pretrain模型，同时都选用Resnet50作为backbone

2、草图数据库包含25个类别，每个类别800张草图，一共20，000张草图。20个类别作为seen类别，5个类别作为unseen类别，每个类别650张草图作为训练集，50张作为验证集，100张作为测试集

3、草图有2种格式，png格式（分辨率256x256），可以点击[这里](https://pan.baidu.com/s/1DRawrYLnBmxXe3laDSiBCA?pwd=94cu) 从百度网盘下载PNG格式数据,svg格式（包含笔画顺序信息等），可以点击[这里](https://pan.baidu.com/s/1kYSV-GFOgIR0U2tFpgKD9Q?pwd=4fsp)从百度网盘下载SVG格式数据。

4、训练集和验证集按照类别放在不同文件夹内，测试集数据分为2种，一种是seen，即这些数据类别在训练集中出现过，所有seen的测试数据放在一个文件夹内，一种是unseen类别，即这些数据类别未在训练集中出现过，所有unseen的测试数据放在一个文件夹内。测试集的数据标注分别在test_seen_label.txt和test_unseen_label.txt中。

5、采用F1作为评价指标。利用SRC_Evaluation.py中的代码进行评测 (感谢李广信和林康衡帮忙写评测代码)，使用示例如下
需要安装pandas，openpyxl,scikit-learnd库

```python
    img_names = ['19142.png', '19143.png', '19144.png', '19145.png'] ##测试图片名，利用list存储
    pred_labels = ['mouse', 'backpack', 'backpack', 'backpack']      ##预测类别名，和上面的图片名一一对应，利用list存储
    pred_file = write2excel(img_names, pred_labels)   ##输入图片名和类别名，将其存储在excel文件中,并返回文件名
    evaluate(pred_file,seen=1)   ##将上面的文件名和测试集类型，获得评价结果。测试集类型分为seen和unseen，默认为1，为seen类别，设置为0，为unseen类别
```

# 草图识别挑战-TUBerlin

1、数据来源于[TUBerlin草图数据库](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)，包含250个类别，每个类别80张草图，一共20，000张草图。我将每个类别中50张作为训练集，30张作为测试集，详细信息见数据库train.txt 和 test.txt.可以点击[这里](https://pan.baidu.com/s/17Ja1LySQprZJhs8Vk1YNFQ?pwd=twvf)从百度网盘下载。

2、采用F1作为评价指标。

# 建议处理顺序
1、下载数据库，查看数据存放格式

2、撰写数据加载器

3、加载模型，初始化模型，创建优化器和损失函数

4、加在数据，训练模型

5、在验证集上验证模型

6、保留最优的模型

7、最优模型在测试集上测试，利用SRC_Evaluation.py中的代码进行评测。记录结果

8、继续调整参数，获得最优模型并进行评测

9、将最新结果汇报给我，我给添加到github

# baseline代码说明
我上传了一份可以跑这两个任务的baseline代码。SPG大概能在0.97以上(seen)。TUBerlin大概能在0.77以上。由于我没有完全跑完，所以结果仅作参考。

提升精度的方法有：

1、更换pretrain模型，现在有很多在大量数据集上训练无监督方法，这个模型是不是比现在的pretrain模型好？这些常见的方法有(DeepClusterv2)[https://github.com/facebookresearch/swav], (SWAV)[https://github.com/facebookresearch/swav], (OBOW)[https://github.com/valeoai/obow],(BarlowTwins)[https://github.com/facebookresearch/barlowtwins]等等








