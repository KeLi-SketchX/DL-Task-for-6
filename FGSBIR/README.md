# 细粒度的基于草图的图像检索

当你想进行精细的检索时，草图是不二之选。（1）“一图胜千言”，草图能够本质，自然，清晰的且无歧义表达人们心中所想；（2）绘画是人本身就具备的能力，能跨越语言和地区；（3）随着触屏式设备的普及，草图的获取非常方便

目前细粒度的基于草图的图像检索的经典框架是利用CNN提取特征，利用特征相似度进行排序检索，训练时利用triplet loss约束草图和图像间的特征空间。

[数据库](https://pan.baidu.com/s/1b1W1JnzjIZcOeMMdip-BLg?pwd=qn5v)。数据库的图片存放格式如下所示。草图和图片的对应关系见sketch_photo_id.mat
```
-train
  -photo
    1.png
    2.png
    ...
  -sketch
    1.png
    2.png
    ...
-test
  -photo
    1.png
    2.png
    ...
  -sketch
    1.png
    2.png
    ...
```
sketch_photo_id.mat中字典‘photo_id_train’和‘photo_id_test’分别存储训练集和测试集草图和图片的对应关系，两者格式一样。

以字典‘photo_id_test’为例说明，data[‘photo_id_test’]数据形式为666*1的数组，data[‘photo_id_test’][0]为test/sketch/1.png对应的photo文件名，data[‘photo_id_test’][1]为test/sketch/2.png对应的photo文件名。

# 建议
大家可以直接用resnet50作为backbone。
