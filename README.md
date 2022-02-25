# NTS-Net
```
细粒度分类学习:ECCV2018 paper "Learning to Navigate for Fine-grained Classification"
理解:
论文中提出了三个模型: Navigator, Teacher, Scrytinizer.
Navigator实际上就是RPN，结合了特征金字塔结构FPN, 不同层级需要设定不同的Anchor Box的大小和比例，
这些Anchor Box作为候选区域，Navigator需要预测候选区域对应的信息丰富度，因为这些信息丰富度需要监督，作者采用
自监督的方法，采用一个Teacher网络将这些候选区域规范化后用特征提取器再次提取特征并得到得分，得分通过标签监督，
这些得分可以作为Navigator的监督，要求Navigator预测候选区域对应的信息丰富度排序和Teacher网络预测得分的排序一样，
用到了RANK　LOSS. 论文中说总LOSS是三个LOSS监督，一个是Teacher预测候选区域得分的监督LOSS, 
一个是自监督的Teacher得分对Navigator的反馈监督LOSS，一个是融合特征后的监督LOSS. 
但代码中还是加了粗分类的监督LOSS,代码中还有很多细节，比如TOPK, NMS等等...., 
我对代码进行了优化并添加加了注释，便于交流学习，最后感谢作者的工作。
```
## 基础环境
- torch==1.9.0
- python==3.8.8
- see dockerfile...

## 数据集
```
数据使用standford car, 数据可在官网下载，使用的更新版.
datasets
└──stankford_car
    └── car_ims
    └── stanford_cars_annos.mat
```

## 训练和测试
```
支持数据并行
编译环境：python ./setup.py build develop
训练:python train.py
测试:python test.py
```
## 结果
|模型|Method|Acc@1|
|---|---|---|
|resnet50|细粒度分类|－|
|resnet50|粗分类|－|
|resnet18|细粒度分类|－|
|resnet18|粗分类|－|

## 参考
```
官方链接：https://github.com/yangze0930/NTS-Net
```
