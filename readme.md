博客地址：https://www.cnblogs.com/panweiwei/

GitHub地址：https://github.com/PanWeiw/-CNN-FacesRecognition.git

项目环境：详见requirements.txt文件；

项目结构：![image-20200327123843521](C:\Users\pw\AppData\Roaming\Typora\typora-user-images\image-20200327123843521.png)

注：

dataset文件夹是数据集；

MyModel文件夹存放训练的模型问价；

model_test文件夹是测试模型时，存放待测照片、视频所用；

my_faces和MySource_faces文件夹是测试采集人脸数据集时所用的。

****************************

一：采集人脸数据集
从照片采集：faces_from_Photo.py
从视频文件采集：faces_from_Video.py
从摄像头采集：faces_from_Camera.py

加载照片的函数：read_img.py


二：处理照片并加载标签
load_dataset.py

三：构建并训练模型
train_model.py

处理数据集的函数：dataSet.py

四：模型测试
识别照片中的人脸：Recognize_From_Photo.py
识别视频文件中的人脸：Recognize_From_Vedio.py
识别摄像头中的人脸：Recognize_From_Camera.py