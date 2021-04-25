# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os


def main(args):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.datasets import mnist
    from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.models import Model
    from keras.optimizers import Adam
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

    # construct model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['acc'],
    )
    model.summary()

    # train model
    print('Training...')
    model.fit(x_train, y_train, batch_size=128, epochs=15)

    # evaluate model
    print('Evaluating...')
    result = model.evaluate(x_test, y_test, verbose=0)
    print(f'Loss:     {result[0]:0.3f}')
    print(f'Accuracy: {result[1]:0.3f}')

    # save model
    file_path = 'mnist_dnn.h5'
    print(f'Saving {file_path}...')
    model.save(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
