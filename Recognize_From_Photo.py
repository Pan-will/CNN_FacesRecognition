"""-----------------------------------------
step 4:
    识别照片中的人脸；
-----------------------------------------"""
# -*-coding:utf8-*-
from load_dataset import read_name_list, read_file
from train_model import Model
import cv2

# 测试识别一张图片
def test_onePicture(path):
    # 加载模型
    model = Model()
    model.load()
    # 读取图片
    img = cv2.imread(path)
    # 重置图片尺寸为：128*128
    img = cv2.resize(img, (128, 128))
    # 图片灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # labelIndex为概率最高的label的索引号, prob为对应概率
    labelIndex, prob = model.predict(img)
    if labelIndex != -1:
        name_list = read_name_list('./dataset')
        print(name_list[labelIndex], prob)
    else:
        print("Don't know this person.")


# 读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    # 加载模型
    model = Model()
    model.load()
    # 计数器
    index = 0
    # 读取所有图片；img_list是所有图片的集合，label_lsit是所有标签的集合，label_num是标签数量
    img_list, label_lsit, label_num = read_file(path)
    for img in img_list:
        # labelIndex为概率最高的label的索引号, prob为对应概率
        labelIndex, prob = model.predict(img)
        if labelIndex != -1:
            index += 1
            name_list = read_name_list('./dataset')
            print(name_list[labelIndex])
        else:
            print("Don't know this person.")
    return index


if __name__ == '__main__':
    # test_onePicture('./model_test/panwei1.jpg')
    test_onBatch('./model_test')