# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


def generateSubmodel(x_train, y_train, x_test, y_test):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    print(f"size of xtrain: {len(x_train)}")

    num_classes = 10
    weight_decay = 1e-4
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
               input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # training
    batch_size = 64

    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=125,
                        verbose=1, validation_data=(x_test, y_test), callbacks=[LearningRateScheduler(lr_schedule)])
    return model

def generateModel(ensembleSize):
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    import math

    num_classes = 10
    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # scale images from 0-1 to 0-255 and add channel dimension
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #
    # # z-score
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    submodels = []
    """
    x mutually exclusive training set.
    """
    for i in range(ensembleSize):
        print(f"Training a submodel using training set [{math.floor(i*len(x_train)/ensembleSize)} : {math.floor((i+1)*len(x_train)/ensembleSize)}]")
        print(math.floor(i*len(x_train)/ensembleSize))
        smodel = generateSubmodel(x_train[math.floor(i*len(x_train)/ensembleSize):math.floor((i+1)*len(x_train)/ensembleSize)], y_train[math.floor(i*len(y_train)/ensembleSize):math.floor((i+1)*len(y_train)/ensembleSize)], x_test, y_test)
        submodels.append(smodel)
        # save model
        file_path = 'hetero08232021/dnn_ensemble_target_' + str(i) + '.h5'
        print(f'Saving {file_path}...')
        smodel.save(file_path)

    model = getEnsemble(ensembleSize)
    model.evaluate(x_test, y_test)


"""
submodels are trained with more training samples, but may overlap with other submodel's training data.
"""
def generateNonMutualExclusiveModel(ensembleSize):
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    import math

    num_classes = 10
    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # scale images from 0-1 to 0-255 and add channel dimension
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #
    # # z-score
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    i = 4
    if i == ensembleSize-1:
        # print("adding head to tail")
        print(f"Training a submodel using training set [{math.floor(i*len(x_train)/ensembleSize)} : {math.floor((i+1)*len(x_train)/ensembleSize)}] and set 0 - {math.floor((1)*len(x_train)/ensembleSize)}")
        x_training = np.concatenate((x_train[math.floor(i*len(x_train)/ensembleSize):math.floor((i+1)*len(x_train)/ensembleSize)], x_train[0:math.floor(1*len(x_train)/ensembleSize)]))
        y_training = np.concatenate((y_train[math.floor(i*len(y_train)/ensembleSize):math.floor((i+1)*len(y_train)/ensembleSize)], y_train[0:math.floor(1*len(y_train)/ensembleSize)]))
        # print(len(x_training))
        # print(len(y_training))
        smodel = generateSubmodel(x_training, y_training, x_test, y_test)
        # save model
        file_path = 'higherAccHomo08152021/dnn_ensemble_target_' + str(i) + '.h5'
        print(f'Saving {file_path}...')
        smodel.save(file_path)

    # submodels = []
    # for i in range(ensembleSize):
    #     i = 4
    #     if i == ensembleSize - 1:
    #         # print("adding head to tail")
    #         print(
    #             f"Training a submodel using training set [{math.floor(i * len(x_train) / ensembleSize)} : {math.floor((i + 1) * len(x_train) / ensembleSize)}] and set 0 - {math.floor((1) * len(x_train) / ensembleSize)}")
    #         x_training = np.concatenate((x_train[math.floor(i * len(x_train) / ensembleSize):math.floor(
    #             (i + 1) * len(x_train) / ensembleSize)], x_train[0:math.floor(1 * len(x_train) / ensembleSize)]))
    #         y_training = np.concatenate((y_train[math.floor(i * len(y_train) / ensembleSize):math.floor(
    #             (i + 1) * len(y_train) / ensembleSize)], y_train[0:math.floor(1 * len(y_train) / ensembleSize)]))
    #         # print(len(x_training))
    #         # print(len(y_training))
    #         smodel = generateSubmodel(x_training, y_training, x_test, y_test)
    #         submodels.append(smodel)
    #         # save model
    #         file_path = 'higherAccHomo08152021/dnn_ensemble_target_' + str(i) + '.h5'
    #         print(f'Saving {file_path}...')
    #         smodel.save(file_path)
    #     else:
    #         print("i and i+1")
    #         print(
    #             f"Training a submodel using training set [{math.floor(i * len(x_train) / ensembleSize)} : {math.floor((i + 2) * len(x_train) / ensembleSize)}]")
    #         smodel = generateSubmodel(x_train[math.floor(i * len(x_train) / ensembleSize):math.floor(
    #             (i + 2) * len(x_train) / ensembleSize)], y_train[math.floor(i * len(y_train) / ensembleSize):math.floor(
    #             (i + 2) * len(y_train) / ensembleSize)], x_test, y_test)
    #         submodels.append(smodel)
    #         # save model
    #         file_path = 'higherAccHomo08152021/dnn_ensemble_target_' + str(i) + '.h5'
    #         print(f'Saving {file_path}...')
    #         smodel.save(file_path)


    model = getEnsemble(ensembleSize)
    model.evaluate(x_test, y_test)


class Ensemble:
    mean = 120.70756512369792
    std = 64.1500758911213

    def __init__(self):
        self.submodels = []

    def append(self, submodel):
        self.submodels.append(submodel)

    def predict(self, x):
        # x = (x - self.mean) / (self.std + 1e-7)

        result = np.zeros(10)
        for i in range(len(self.submodels)):
            temp = self.submodels[i].predict(x)
            result = [a + b for a, b in zip(result, temp)]
        return np.true_divide(result, len(self.submodels))

    def evaluate(self, x_test, y_test):
        num_classes = 10

        correctCount = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, 0)
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correctCount += 1
        total = len(x_test)
        print("Out of " + str(total) + " test samples, the ensemble method correctly predicts " + str(correctCount) + " (" + str(correctCount/total) + ")")

    def recursive_evaluate(self, x_test, y_test):
        print(f"\nEvaluating each submodel's accuracy")
        for i in range(len(self.submodels)):
            correctCount = 0
            for x, y in zip(x_test, y_test):
                x = np.expand_dims(x, 0)
                prediction = self.submodels[i].predict(x)
                if np.argmax(prediction) == np.argmax(y):
                    correctCount += 1
            total = len(x_test)
            print("Out of " + str(total) + " test samples, the submodel {" + str(i) + "} correctly predicts " + str(
                correctCount) + " (" + str(correctCount / total) + ")")


def getEnsemble(ensembleSize):
    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import load_model

    model = Ensemble()
    for i in range(ensembleSize):
        file_path = 'hetero08232021/dnn_ensemble_target_' + str(i) + '.h5'
        print(f'Loading {file_path}...')
        submodel = load_model(file_path)
        model.append(submodel)

    return model


def main(args):
    num_classes = 10
    input_shape = (32, 32, 3)

    from keras.datasets import cifar10
    from keras.utils import to_categorical
    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)


    # load CIFAR-10 dataset
    print(f'Input shape: {x_train.shape}')

    # # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')
    generateModel(3)
    # generateNonMutualExclusiveModel(5)
    # evaluate model
    model = getEnsemble(3)
    print('Evaluating...')

    model.evaluate(x_test, y_test)
    model.recursive_evaluate(x_test, y_test)


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
