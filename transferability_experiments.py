# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from PIL import Image


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
    from keras.applications.resnet50 import preprocess_input, decode_predictions

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

    file_path = 'cifar10_dnn.h5'
    print(f'Loading {file_path}...')
    model2 = load_model(file_path)
    # model.summary()

    # # demonstrate
    # x = np.expand_dims(x_test[333], 0)
    # y = model.predict(x)
    # print('Displaying input...')
    # img = Image.fromarray(np.uint8(x.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()
    # print(f'Prediction: {y}')
    # print(f'Prediction: {class_labels[np.argmax(y)]}')
    import _pickle as cPickle
    with open(r"adv.pickle", "rb") as input_file:
        x = cPickle.load(input_file)
        x = np.expand_dims(x, axis=0)  # x: [1, 28, 28, 1]

        #COMMENTED
    # img = Image.open('adv.png')
    # img.show()
    # img = img.convert('RGB')
    # img = img.resize((32, 32))
    # img.show()
    # # x = np.float32(img) / 255
    # x = np.asarray(img)
    # # x = np.mean(x, axis=-1)  # x: [28, 28]
    # # x = transform.resize(x, (256, 256, 3))
    # x = np.expand_dims(x, axis=0)  # x: [1, 28, 28, 1]
    # print(x)
    # # img_preprocessed = preprocess_input(x)


    y = model.predict(x)
    print(f'shape: {x.shape}')
    print(f'Prediction: {y}')
    print(f'Prediction: {np.argmax(y)}')
    print(f'Prediction: {class_labels[np.argmax(y)]}')

    print("Regular")
    y2 = model2.predict(x)
    print(f'Prediction: {y2}')
    print(f'Prediction: {np.argmax(y2)}')
    print(f'Prediction: {class_labels[np.argmax(y2)]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
