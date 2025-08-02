# Traffic-detection-based-on-YOLOv8

VisDrone2019数据集下载：https://github.com/VisDrone/VisDrone-Dataset
注意：请下载网页中的“Task 2: Object Detection in Videos”内容！

请将下载的文件解压到程序根目录中的\data\VisDrone2019文件夹中，运行convert.py程序转换为YOLO格式（请记得调整程序第7行中的数据集保存路径），转换好的格式会保存在\data\YOLO_Dataset文件夹中

由于数据量较大，训练模型时推荐使用cuda加速
训练完成的模型会保存在\runs\train\visdrone_v8s\weights文件夹中，需要将其中的best.pt模型另存为VD_s.pt保存在项目根目录中使用！

首次运行main.py程序可能加载时间较长，请耐心等待（之后就不会了）
可自行选取测试视频文件进行测试
