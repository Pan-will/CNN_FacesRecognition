"""-----------------------------------------
step 1：从视频文件采集人脸数据；
        一次一个label；
        人脸大小：128*128
-----------------------------------------"""
import cv2
import dlib
import os
import random

# 采集的照片越多、越清晰越好
# faces_my_path文件夹下存放检测以后的照片
faces_my_path = './my_faces'
# 图片大小
size = 128
# 若目录不存在，则先创建目录
if not os.path.exists(faces_my_path):
    os.makedirs(faces_my_path)


"""改变图片的相关参数：亮度与对比度"""
def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, k] = tmp
    return img


# 特征提取器:dlib自带的frontal_face_detector,即正脸检测器
detector = dlib.get_frontal_face_detector()

#创建窗口
cv2.namedWindow('Collecting photos', 1)
# 调整窗口大小
cv2.resizeWindow('Collecting photos', 400, 300)
# 打开摄像头；内置摄像头为0，若有其他摄像头则依次为1,2,3,4...
camera = cv2.VideoCapture('collect_faces1.mp4')

if False == camera.isOpened():
    print("未能打开视频文件。")
else:
    print("视频文件已打开！")

# 计数器
index = 1
# 我的人脸数据集容量
datasetSize = 200
while True:
    if (index <= datasetSize):
        success, img = camera.read()
        if success:
            print("视频文件读取图像成功！", index)
        else:
            print("未能通过视频文件读取图像！")

        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)
        """--------------------------------------------------------------------
        使用enumerate 函数遍历序列中的元素以及它们的下标,i为人脸序号,d为i对应的元素;
        left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
        top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        ----------------------------------------------------------------------"""
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            # 调整图片的亮度与对比度， 亮度与对比度值都取随机数，这样能增加样本的多样性
            face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            face = cv2.resize(face, (size, size))

            cv2.imshow('Collecting...', face)

            cv2.imwrite(faces_my_path + '/' + str(index) + '.jpg', face)
            index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(datasetSize, "张人脸图像采集完成。")
        break
