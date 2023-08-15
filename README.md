<<<<<<< HEAD
# CropUnet

基于Unet网络的无人机全景图的耕地检测模型

主要分为两个部分：

一、训练模型，获得模型权重；


二、打包模型，部署环境；


=======
Unet模型适合特征少，需要浅层特征的全景农田数据集

![图片1(phone/18_px_0_1024.png "Magic Gardens")

![](RackMultipart20230811-1-hsq30t_html_fb7ab48ce863e255.png) ![](RackMultipart20230811-1-hsq30t_html_4e2bb639dfbbc09a.png)

1. 权重文件下载

训练所需的权值可在百度网盘中下载，下载后放到model。

链接: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A

提取码: 6n2c

1. 模型训练和测试

1. 将存放全景图像的panorama文件夹下所有全景图切割并提取主要绿地作为农田图像；
2. 训练前将图片文件放在Datasets文件夹下的JPEGImages中;
3. 训练前将标签文件放在Datasets文件夹下的SegmentationClass中;
4. 在训练前利用annotation.py文件生成对应的txt;
5. 修改train.py的num\_classes为分类个数+1，这里只有农田类别，因此该参数设置为1
6. 运行train.py即可开始训练；
7. 修改unet.py中的model\_path和num\_classes，打开predicet.py根据测试模型更改参数，进行预测测试

1. 参考资料

https://github.com/bubbliiiing/unet-pytorch

