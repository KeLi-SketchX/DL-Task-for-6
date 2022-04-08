# 草图识别调整

1、数据来源于SPG-dataset中的草图，是Quick Draw的一个子集，因此为了公平，不使用任何和Quick Draw相关的数据和pretrain模型，同时都选用Resnet50作为backbone

2、草图数据库包含25个类别，每个类别800张草图，一共20，000张草图。20个类别作为seen类别，5个类别作为unseen类别，每个类别650张草图作为训练集，50张作为验证集，100张作为测试集

3、草图有2中存储方式，png格式（分辨率256x256），可以点击[这里](https://drive.google.com/file/d/1wKCvZMmjTnOFiVFEk5v53SrHt_w4Y5-g/view?usp=sharing)从Google Drive下载PNG格式数据,svg格式（包含笔画顺序信息等），可以点击[这里](https://drive.google.com/file/d/1ssCinsxje1f_aGryJVXFkFsTBDFXgJcH/view?usp=sharing)从Google Drive下载SVG格式数据。

4、评价指标
