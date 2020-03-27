"""-----------------------------------------
step 4:
    识别照片中的人脸；
-----------------------------------------"""
# -*-coding:utf8-*-
import cv2
from load_dataset import read_name_list
from train_model import Model

# 待测试图片路径
path = './model_test/yeye.jpg'
# path = './model_test/panwei/panwei1.jpg'
# path = './model_test/dandan/lidan.jpg'
# 获取人脸特征模型
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# 加载模型
model = Model()
model.load()
# 读取图片
image = cv2.imread(path)

# 重置图片尺寸为：128*128、512*512
small_image = cv2.resize(image, (128, 128))

# 图片灰度化
small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸，探测图片中的人脸
faces = face_engine.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=15)

# 读取dataset数据集下的子文件夹名称
name_list = read_name_list('./dataset')
# 框出人脸
for (x, y, w, h) in faces:
    origin = gray[x:x + w, y:y + h]
    # origin是原图，self.img_size是输出图像的尺寸，interpolation是插值方法
    origin = cv2.resize(origin, (128, 128), interpolation=cv2.INTER_LINEAR)
    # 利用模型对cv2识别出的人脸进行比对
    # 第一个返回值为概率最高的label的index,第二个返回值为对应概率
    labelIndex, prob = model.predict(origin)
    # 如果模型认为概率高于70%则显示为模型中已有的label
    if prob > 0.7:
        show_name = name_list[labelIndex]
    else:
        show_name = 'unknown'
    # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
    frame = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # 显示人名
    cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


# 显示图片
cv2.imshow("Recognizing...", image)

# 暂停窗口
cv2.waitKey(5000)
# 销毁窗口
cv2.destroyAllWindows()

# 需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
# labelIndex为概率最高的label的索引号, prob为对应概率
labelIndex, prob = model.predict(small_gray)
if labelIndex != -1:
    name_list = read_name_list('./dataset')
    print(name_list[labelIndex], prob)
else:
    print("Don't know this person.")



