# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from PIL import Image
import AdvGenerator
import random
import math
from Candidate import Candidate
from AdversarialExample import AdversarialExample
from operator import itemgetter, attrgetter


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

    # load models, target DNN model to attack
    file_path = 'cifar10_dnn.h5'
    print(f'Loading {file_path}...')
    model = load_model(file_path)

    # models to test for transferability
    file_path = 'alexnetmodel.h5'
    print(f'Loading {file_path}...')
    model1 = load_model(file_path)

    file_path = 'cifar10_dnn_target.h5'
    print(f'Loading {file_path}...')
    alexnet = load_model(file_path)
    # model.summary()

    starting_index = 2019

    resnetTransCount = 0
    alexnetTransCount = 0
    total = 0
    i = 0
    # for i in range(5):
    while total <= 30:
        i += 1

        print("Testing transferability property for testset[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))
        testmodelprediction = np.argmax(model1.predict(x))
        alexprediction = np.argmax(alexnet.predict(x))

        # print(f'groundtruth: {groundtruth}')
        # print(f'basemodelprediction: {basemodelprediction}')
        # print(f'testmodelprediction: {testmodelprediction}')
        # print(f'alexprediction: {alexprediction}')

        if groundtruth == basemodelprediction and groundtruth == testmodelprediction and groundtruth == alexprediction:

            # generate adversarial example here.
            total += 1

            advExample, generation = AdvGenerator.parallelGAhighConf(populationSize=15, generation=1000, inputImage=image, model=model,
                                                             y_truth=groundtruth, IMAGEDIMENSION=32)

            img = Image.fromarray(np.uint8(advExample.reshape(input_shape) * 255), 'RGB').resize(
                (128, 128))

            img.show(title="test")

            advExample = np.expand_dims(advExample, 0)
            print(f'fail prediction: {model.predict(advExample)}')

            if np.argmax(model.predict(advExample)) == np.argmax(model1.predict(advExample)):
                resnetTransCount += 1

            if np.argmax(model.predict(advExample)) == np.argmax(alexnet.predict(advExample)):
                alexnetTransCount += 1

            print("\nadv example ")
            print(f'groundtruthMOD: {groundtruth}')
            print(f'basemodelpredictionMOD: {model.predict(advExample)}')
            print(f'testmodelpredictionMOD: {model1.predict(advExample)}')
            print(f'alexpredictionMOD: {alexnet.predict(advExample)}')

        print("\n\n\n")

    print(f'Total number of images compared: {total}')
    print(f'Number of adversarial examples transferred to second resnet: {resnetTransCount}')
    print(f'Percentage: {resnetTransCount/total}')
    print(f'Number of adversarial examples transferred : {alexnetTransCount}')
    print(f'Percentage: {alexnetTransCount/total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
