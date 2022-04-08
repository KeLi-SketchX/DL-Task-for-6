# 草图识别调整

1、数据来源于SPG-dataset中的草图，是Quick Draw的一个子集，因此为了公平，不使用任何和Quick Draw相关的数据和pretrain模型，同时都选用Resnet50作为backbone

2、草图数据库包含25个类别，每个类别800张草图，一共20，000张草图。20个类别作为seen类别，5个类别作为unseen类别，每个类别650张草图作为训练集，50张作为验证集，100张作为测试集

3、草图有2种格式，png格式（分辨率256x256），可以点击[这里](https://pan.baidu.com/s/1DRawrYLnBmxXe3laDSiBCA?pwd=94cu) 从百度下载PNG格式数据,svg格式（包含笔画顺序信息等），可以点击[这里](https://pan.baidu.com/s/1kYSV-GFOgIR0U2tFpgKD9Q?pwd=4fsp)从百度网盘下载SVG格式数据。

4、训练集和验证集按照类别放在不同文件夹内，测试集数据分为2种，一种是seen，即这些数据类别在训练集中出现过，所有seen的测试数据放在一个文件夹内，一种是unseen类别，即这些数据类别未在训练集中出现过，所有unseen的测试数据放在一个文件夹内。测试集的数据标注分别在test_seen_label.txt和test_unseen_label.txt中。

5、采用F1作为评价指标
