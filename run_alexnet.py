# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv3D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, history2, history3, history4):
	plt.subplot(2,2,1)
	plt.plot(history.history['val_acc'])
	plt.plot(history2.history['val_acc'])
	plt.plot(history3.history['val_acc'])
	plt.plot(history4.history['val_acc'])

	plt.xticks(np.arange(0,epochs, (epochs/10)))
	plt.title('Val accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,2)
	plt.plot(history.history['val_loss'])
	plt.plot(history2.history['val_loss'])
	plt.plot(history3.history['val_loss'])
	plt.plot(history4.history['val_loss'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Val loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,3)
	plt.plot(history.history['acc'])
	plt.plot(history2.history['acc'])
	plt.plot(history3.history['acc'])
	plt.plot(history4.history['acc'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Train accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,4)
	plt.plot(history.history['loss'])
	plt.plot(history2.history['loss'])
	plt.plot(history3.history['loss'])
	plt.plot(history4.history['loss'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Train loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.show()


def main(args):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.datasets import cifar10
    from keras.models import load_model
    from keras.utils import to_categorical

    input_shape = (32, 32, 3)
    num_classes = 10
    class_labels = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # scale images from 0-255 to 0-1 and add channel dimension
    x_train = np.float32(x_train) / 255
    x_test = np.float32(x_test) / 255
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    # load model
    file_path = 'alexnetmodel.h5'
    print(f'Loading {file_path}...')
    model = load_model(file_path)
    model.summary()

    # demonstrate
    x = np.expand_dims(x_test[2021], 0)
    y = model.predict(x)
    print('Displaying input...')
    img = Image.fromarray(np.uint8(x.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    img.show()
    print(f'Prediction: {y}')
    print(f'Prediction: {class_labels[np.argmax(y)]}')

    # finally:
    print("results: -----")
    # cifar = tf.keras.datasets.cifar10
    # (x_train, y_train), (x_test, y_test) = cifar.load_data()
    #
    # x_train, x_test = x_train / 255.0, x_test / 255.0  # To make the data between 0~1
    # y_test_label = y_test
    # y_train = to_categorical(y_train, num_classes=10)
    # y_test = to_categorical(y_test, num_classes=10)
    #
    # x_train, x_validation = x_train[0:40000], x_train[40000:50000]
    # y_train, y_validation = y_train[0:40000], y_train[40000:50000]
    #
    # print(x_train.shape, 'train samples')
    # print(y_train.shape, 'train labels')
    # print(x_validation.shape, 'validation samples')
    # print(y_validation.shape, 'validation labels')
    # print(x_test.shape, 'test samples')
    # print(y_test.shape, 'train labels')
    # print(x_train[0].shape)
    #
    #
    # score = model.evaluate(x_test, y_test)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # predict_classes = model.predict_classes(x_test)
    #
    #
    # plt.figure(figsize=(12,9))
    # for j in range(0,3):
    # 	for i in range(0,4):
    # 		plt.subplot(4,3,j*4+i+1)
    # 		plt.title('predict:{}/real:{}'.format(predict_classes[j*4+i] ,y_test_label[j*4+i]))
    # 		plt.axis('off')
    # 		plt.imshow(x_test[j*4+i].reshape(32,32,3),cmap=plt.cm.binary)

    # plt.show()

    # # load model
    # print("Testing results for set 0")
    # file_path = 'alexnetmodel.h5'
    # print(f'Loading {file_path}...')
    # model = load_model(file_path)
    # # model.summary()
    #
    # score = model.evaluate(x_test, y_test)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # # predict_classes = model.predict_classes(x_test)
    #
    # # load model
    # print("Testing results for set 2")
    # file_path = 'alexnetmodel2.h5'
    # print(f'Loading {file_path}...')
    # model = load_model(file_path)
    # model.summary()
    #
    # score = model.evaluate(x_test, y_test)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # # predict_classes = model.predict_classes(x_test)
    #
    # # load model
    # print("Testing results for set 3")
    # file_path = 'alexnetmodel3.h5'
    # print(f'Loading {file_path}...')
    # model = load_model(file_path)
    # model.summary()
    #
    # score = model.evaluate(x_test, y_test)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # # predict_classes = model.predict_classes(x_test)
    #
    # # load model
    # print("Testing results for set 4")
    # file_path = 'alexnetmodel4.h5'
    # print(f'Loading {file_path}...')
    # model = load_model(file_path)
    # model.summary()
    #
    # score = model.evaluate(x_test, y_test)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # predict_classes = model.predict_classes(x_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
