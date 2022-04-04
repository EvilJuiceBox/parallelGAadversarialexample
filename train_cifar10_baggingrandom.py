# -*- coding: utf-8 -*-

import argparse
import random

import numpy as np
import os
from random import sample

def generateSubmodel(x_train, y_train, x_test, y_test):
    """
    generates a submodel of ResNet-20 architecture given the training and test data. (test data is used to provided accuracy metrics only)
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: returns a keras.model
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    from keras.layers import Add, Activation, Conv2D, Dense, GlobalAveragePooling2D, Input
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l2

    num_classes = 10
    input_shape = (32, 32, 3)
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



    # construct resnet-20 model
    inputs = Input(shape=input_shape)
    x = inputs
    x = _conv2d_block(16, activation='relu', kernel_size=3, strides=1)(x)

    # construct resnet-20 model: stage 1, block 1
    x_branch = x
    x_branch = _conv2d_block(16, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(16, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 1, block 2
    x_branch = x
    x_branch = _conv2d_block(16, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(16, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 1, block 3
    x_branch = x
    x_branch = _conv2d_block(16, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(16, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 2, block 1
    x_branch = x
    x_branch = _conv2d_block(32, activation='relu', kernel_size=3, strides=2)(x_branch)
    x_branch = _conv2d_block(32, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Conv2D(32, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 2, block 2
    x_branch = x
    x_branch = _conv2d_block(32, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(32, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 2, block 3
    x_branch = x
    x_branch = _conv2d_block(32, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(32, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 3, block 1
    x_branch = x
    x_branch = _conv2d_block(64, activation='relu', kernel_size=3, strides=2)(x_branch)
    x_branch = _conv2d_block(64, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Conv2D(64, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 3, block 2
    x_branch = x
    x_branch = _conv2d_block(64, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(64, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: stage 3, block 3
    x_branch = x
    x_branch = _conv2d_block(64, activation='relu', kernel_size=3, strides=1)(x_branch)
    x_branch = _conv2d_block(64, activation=None, kernel_size=3, strides=1)(x_branch)
    x = Add()([x, x_branch])
    x = Activation('relu')(x)

    # construct resnet-20 model: finalize
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, kernel_initializer='he_normal')(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['acc'],
    )
    model.summary()

    # create learning rate plateau handler
    lr_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=5,
        min_lr=5e-6
    )

    # create learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        lambda x:
        1e-3 * 5e-4 if x > 180
        else 1e-3 * 1e-3 if x > 160
        else 1e-3 * 1e-2 if x > 120
        else 1e-3 * 1e-1 if x > 80
        else 1e-3
    )

    # train model
    print('Training...')
    model.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=[lr_plateau, lr_scheduler])

    # evaluate model
    print('Evaluating...')
    result = model.evaluate(x_test, y_test, verbose=0)
    print(f'Loss:     {result[0]:0.3f}')
    print(f'Accuracy: {result[1]:0.3f}')

    return model

def generateModel(ensembleSize):
    """
    Generates an ensemble model of given size. Uses generateSubmodel function.
    :param ensembleSize:
    :return:
    """
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    import math

    num_classes = 10
    input_shape = (32, 32, 3)


    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # scale images from 0-1 to 0-255 and add channel dimension
    x_train = np.float32(x_train) / 255
    x_test = np.float32(x_test) / 255
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    print("pregeneration model")
    submodels = []
    for i in range(ensembleSize):
        # print("getting all other bags")
        # # contains random training data from x or y such that x_temp and y_temp does not contain the data in the training bag of submodel i
        # x_temp = [x for x in x_train if x not in x_train[math.floor(i*len(x_train)/ensembleSize):math.floor((i+1)*len(x_train)/ensembleSize)]]
        # y_temp = [y for y in y_train if y not in y_train[math.floor(i*len(y_train)/ensembleSize):math.floor((i+1)*len(y_train)/ensembleSize)]]

        print("getting additional samples")
        # # randomly sample the other bags to obtain more training data
        # x_sub, y_sub = zip(*random.sample(list(zip(x_train, y_train)), math.floor(len(x_train)/2)))
        # x_sub, y_sub = zip(sample(list(zip(x_temp, y_temp)), math.floor(len(x_train)/ensembleSize)))

        print("generating model")
        # print(f"subtitle{len(x_sub)}")
        # each ensemble is trained with its own bag, and some additional training data.
        smodel = generateSubmodel(x_train[math.floor(i*len(x_train)/ensembleSize):math.floor((i+1)*len(x_train)/ensembleSize)], y_train[math.floor(i*len(y_train)/ensembleSize):math.floor((i+1)*len(y_train)/ensembleSize)], x_test, y_test)
        # smodel = generateSubmodel(x_train[math.floor(i*len(x_train)/ensembleSize):math.floor((i+1)*len(x_train)/ensembleSize)] + x_sub, y_train[math.floor(i*len(y_train)/ensembleSize):math.floor((i+1)*len(y_train)/ensembleSize)] + y_sub, x_test, y_test)
        # smodel = generateSubmodel(x_sub, y_sub, x_test, y_test)
        submodels.append(smodel)
        # save model
        file_path = 'ensemble/cifar10_ensemble_target_' + str(i) + '.h5'
        print(f'Saving {file_path}...')
        smodel.save(file_path)


class Ensemble:
    """
    Ensemble data structure class used to allow for submodel prediction, evaluation, etc.
    """
    def __init__(self):
        self.submodels = []

    def append(self, submodel):
        self.submodels.append(submodel)

    def predict(self, x):
        result = np.zeros(10)
        # print("length of submodel: " + str(len(self.submodels)))
        for i in range(len(self.submodels)):
            temp = self.submodels[i].predict(x)
            result = [a + b for a, b in zip(result, temp)]
            # print("submodel prediction: " + str(result))
        return np.true_divide(result, len(self.submodels))

    def evaluate(self, x_test, y_test):
        correctCount = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, 0)
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correctCount += 1
        total = len(x_test)
        print("Out of " + str(total) + " test samples, the ensemble method correctly predicts " + str(correctCount) + " (" + str(correctCount/total) + ")")


def getEnsemble(ensembleSize):
    """
    loads an ensemble stored in system storage
    returns an Ensemble object as defined in this file.
    """
    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import load_model

    model = Ensemble()
    for i in range(ensembleSize):
        file_path = 'ensemble/cifar10_ensemble_target_' + str(i) + '.h5'
        print(f'Loading {file_path}...')
        submodel = load_model(file_path)
        model.append(submodel)

    return model # Ensemble object


def main(args):
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    import math
    from keras.models import load_model

    num_classes = 10
    input_shape = (32, 32, 3)


    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    # scale images from 0-1 to 0-255 and add channel dimension
    x_train = np.float32(x_train) / 255
    x_test = np.float32(x_test) / 255
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    generateModel(3)

    print("ensemble methods")
    # evaluate model
    # model = load_model("ensemble/cifar10_ensemble_target_0.h5")
    model = getEnsemble(5)
    print('Evaluating...')
    # result = model.predict(np.expand_dims(x_train[2020], 0))
    # # print(result)
    # print(y_train[2020])
    # # print(f'Loss:     {result[0]:0.3f}')
    # # print(f'Accuracy: {result[1]:0.3f}')
    #
    # x = np.expand_dims(x_train[2020], 0)
    # print(model.predict(x))
    # print(y_train[2020])

    model.evaluate(x_test, y_test)

class _conv2d_block(object):
    def __init__(self, num_filters, kernel_size=3, strides=1, activation=None, batch_norm=True):
        from keras.layers import Activation, BatchNormalization, Conv2D
        from keras.regularizers import l2
        self._layers = list()
        self._layers.append(
            Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4),
                   )
        )
        if batch_norm:
            self._layers.append(BatchNormalization())
        if activation is not None:
            self._layers.append(Activation(activation))

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        for layer in self._layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
