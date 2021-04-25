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

    from keras.datasets import mnist
    from keras.models import load_model
    from keras.utils import to_categorical

    input_shape = (28, 28, 1)
    num_classes = 10

    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # scale images from 0-1 to 0-255 and add channel dimension
    x_train = np.expand_dims(np.float32(x_train) / 255, -1)
    x_test = np.expand_dims(np.float32(x_test) / 255, -1)
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    # load model
    file_path = 'mnist_dnn.h5'
    print(f'Loading {file_path}...')
    model = load_model(file_path)
    model.summary()

    # demonstrate
    """
    x = np.expand_dims(x_test[0], 0)
    y = model.predict(x)
    print('Displaying input...')
    img = Image.fromarray(np.uint8(x.reshape(input_shape[:-1]) * 255), 'L').resize((128, 128))
    img.show()
    print(f'Prediction: {y}')
    print(f'Prediction: {np.argmax(y)}')
    """

    img = Image.open('example_2.jpeg')
    img.show()
    img = img.resize((28, 28))
    x = np.float32(img) / 255
    x = np.mean(x, axis=-1)     # x: [28, 28]
    x = np.expand_dims(x, -1)   # x: [28, 28, 1]
    x = np.expand_dims(x, 0)    # x: [1, 28, 28, 1]
    y = model.predict(x)
    print(f'Prediction: {y}')
    print(f'Prediction: {np.argmax(y)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
