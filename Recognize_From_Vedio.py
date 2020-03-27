"""-----------------------------------------
step 4：
    识别视频文件中的人脸；
-----------------------------------------"""
# -*- coding:utf-8 -*-
import cv2
from train_model import Model
from load_dataset import read_name_list

class Camera_reader(object):
    # 在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128

    def build_camera(self):
        # opencv文件中人脸级联文件的位置，用于帮助识别图像或者视频流中的人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        # 读取dataset数据集下的子文件夹名称，即标签名
        name_list = read_name_list('./dataset')

        # 打开摄像头并开始读取画面
        cameraCapture = cv2.VideoCapture('recognize_faces2.mp4')
        success, frame = cameraCapture.read()

        while success and cv2.waitKey(1) == -1:
            success, frame = cameraCapture.read()
            # 图像灰度化
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检测人脸：scaleFactor是放大比例（此参数必须大于1）；minNeighbors是重复识别次数（此参数用来调整精确度，越大则越精确）
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
            for (x, y, w, h) in faces:
                origin = gray[x:x + w, y:y + h]
                # origin是原图，self.img_size是输出图像的尺寸，interpolation是插值方法
                origin = cv2.resize(origin, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                # 利用模型对cv2识别出的人脸进行比对
                label, prob = self.model.predict(origin)
                # 如果模型认为概率高于70%则显示为模型中已有的label
                if prob > 0.7:
                    show_name = name_list[label]
                else:
                    show_name = 'unknown'

                # 框出人脸
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                # 显示人名
                cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            cv2.imshow("Recognizing...", frame)
        # 释放摄像头
        cameraCapture.release()
        # 关闭所有窗口
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = Camera_reader()
    camera.build_camera()
