"""-----------------------------------------
step 1: 从照片中加载人脸数据；
        一次一个label；
        从源文件夹中加载改路径下的所有照片，存储到目标路径中，
    目标路径文件夹的名字是为该批照片的标签；
-----------------------------------------"""
# -*-coding:utf8-*-
import os
import cv2
import time
from read_img import readAllImg

"""
从源路径中读取所有照片,并放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
"""
# sourcePath是存储图像源的文件夹，objectPath是存储识别出的人脸的文件夹，*suffix是源文件的格式
def readPicSaveFace(sourcePath, objectPath, *suffix):
    try:
        # 读取照片,注意第一个元素是文件名
        resultArray = readAllImg(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # 加载人脸特征库
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i) != str:
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    # 以时间戳和读取的排序作为文件名称
                    listStr = [str(int(time.time())), str(count)]
                    print(listStr)
                    fileName = ''.join(listStr)

                    f = cv2.resize(gray[y:(y + h), x:(x + w)], (200, 200))
                    cv2.imwrite(objectPath + os.sep + '%s.jpg' % fileName, f)
                    count += 1

    except IOError:
        print("Error")

    else:
        print('已经读取 ' + str(count - 1) + ' 张照片到 ' + objectPath + '目录。')



if __name__ == '__main__':
    """
    第一个参数：是存储源文件图片的文件夹地址；
    第二个参数：是存储剪裁好、处理好的人脸图片的地址；
    后面的图片格式：源文件图片的格式（文件后缀）。
    """
    # ./dataset/panwei
    readPicSaveFace('./MySource_faces', './my_faces', '.jpg', '.JPG', 'png', 'PNG')
