博客地址：https://www.cnblogs.com/panweiwei/

GitHub地址：https://github.com/PanWeiw/CNN_FacesRecognition.git

## 项目依赖
详见requirements.txt文件；

## 项目结构

dataset：存放数据集；

MyModel：存放训练的模型文件；

model_test：测试模型时，存放待测的照片、视频；

my_faces / MySource_faces：用于测试采集人脸数据集时存放对应数据；

****************************

## 一：采集人脸数据集

从照片采集：faces_from_Photo.py

从视频文件采集：faces_from_Video.py

从摄像头采集：faces_from_Camera.py

加载照片的函数：read_img.py

## 二：处理照片并加载标签

load_dataset.py

## 三：构建并训练模型

train_model.py

处理数据集的函数：dataSet.py

## 四：模型测试

识别照片中的人脸：Recognize_From_Photo.py

识别视频文件中的人脸：Recognize_From_Vedio.py

识别摄像头中的人脸：Recognize_From_Camera.py
