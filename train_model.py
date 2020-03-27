"""-----------------------------------------
step 3：
    构建并训练模型；
-----------------------------------------"""
# -*-coding:utf8-*-
from dataSet import DataSet
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np



# 定义(基于CNN的人脸识别的)模型类
class Model(object):
    # 模型进行存储和读取的路径，以及模型名称
    FILE_PATH = "./MyModel/model.h5"
    # 模型接受的人脸图片大小：128*128
    IMAGE_SIZE = 128

    def __init__(self):
        # 模型初始化
        self.model = None

    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset

    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            # 有很多可选的optimizer，例如RMSprop,Adagrad
            optimizer='adam',
            # 也可以选用squared_hinge作为loss
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=8, batch_size=20)

    # 评估模型
    def evaluate_model(self):
        print('\n............模型评估中............')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('模型损失：', loss)
        print('模型精确度：', accuracy)

    # 保存模型
    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    # 加载模型
    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    # 需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self, img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img / 255.0

        # 计算该img属于某个label的概率
        result = self.model.predict_proba(img)
        # 找出概率最高的
        max_index = np.argmax(result)

        # 第一个返回值为概率最高的label的index,第二个返回值为对应概率
        return max_index, result[0][max_index]


if __name__ == '__main__':
    # 输入数据集路径
    dataset = DataSet('./dataset')
    model = Model()
    # 读取训练集
    model.read_trainData(dataset)
    # 建立模型
    model.build_model()
    # 训练模型
    model.train_model()
    # 评估模型
    model.evaluate_model()
    # 存储模型
    model.save()
