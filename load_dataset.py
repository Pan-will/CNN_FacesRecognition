"""-----------------------------------------
step 2:
    加载数据集中的所有照片和标签；
    函数返回值列表：所有图片集合、标签集合、标签数；
-----------------------------------------"""
# -*-coding:utf8-*-
import os
import cv2
import numpy as np

from read_img import endwith

"""
输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label；
返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)；
如dataset下有多个文件夹，每个文件夹名称对应一个标签（label）；
"""
def read_file(path):
    # 图片集合
    img_list = []
    # 标签集合
    label_list = []
    # 子文件夹个数，即label数
    label_num = 0
    # 图片尺寸
    IMG_SIZE = 128

    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)

        for dir_image in os.listdir(child_path):
            if endwith(dir_image, 'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(label_num)
        # label数自增
        label_num += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    # 返回值列表：所有图片集合、标签集合、标签数
    return img_list, label_list, label_num


# 读取训练数据集下的文件夹，把它们的名称，即标签，返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


if __name__ == '__main__':
    img_list, label_lsit, label_num = read_file('./dataset')
    # 打印一下label数
    print(label_num)
    print(label_lsit)
