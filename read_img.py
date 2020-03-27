"""-----------------------------------------
step 1: 定义加载照片的函数；
        从源文件夹中加载所有照片，照片的格式可由参数*suffix指定，
    返回值是一个list；
-----------------------------------------"""
# -*-coding:utf8-*-
import os
import cv2

"""
根据输入的文件夹路径，将该文件夹下的所有指定suffix的文件读取存入一个list
————该list的第一个元素是该文件夹的名字
"""
def readAllImg(path, *suffix):
    try:
        s = os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)

        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)


    except IOError:
        print("Error")

    else:
        print("读取成功")
        # 返回装有该路径下所有图片的list
        return resultArray


# 输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
# *endstring是文件后缀名：如jpg、png等
def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:
        return True
    else:
        return False

if __name__ == '__main__':
    result = readAllImg("./MySource_faces", '.jpg', '.png')
    print(result[0])