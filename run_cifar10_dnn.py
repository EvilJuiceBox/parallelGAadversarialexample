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
    file_path = 'models/kumardnn.h5'
    print(f'Loading {file_path}...')
    model = load_model(file_path)
    model.summary()

    # z-score
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    print('Evaluating...')
    # result = model.evaluate(x_train, y_train, verbose=0)
    result = model.evaluate(x_test, y_test, verbose=0)
    print("Model metric name: " + str(model.metrics_names))
    print(" ")
    print(result)
    print(f'Loss:     {result[0]:0.3f}')
    print(f'Accuracy: {result[1]:0.3f}')


    # # demonstrate
    # x = np.expand_dims(x_test[333], 0)
    # y = model.predict(x)
    # print('Displaying input...')
    # img = Image.fromarray(np.uint8(x.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()
    # print(f'Prediction: {y}')
    # print(f'Prediction: {class_labels[np.argmax(y)]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
