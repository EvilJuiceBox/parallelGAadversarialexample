import argparse
import os
import time

import numpy as np
from keras.models import load_model

import AdvGenerator
import targetedGeneticAlgorithm
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
from ensembleMain import Result, saveResults


def calculate_distance(filepath, resultfolderpath, image_index=0, number_of_examples=5):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.datasets import cifar10
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

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # scale images from 0-255 to 0-1 and add channel dimension
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')
    cgGA = targetedGeneticAlgorithm.TargetedGeneticAlgorithm()

    image = x_test[image_index]
    ground_truth = y_test[image_index]
    #
    # print(ground_truth)
    # print(np.where(ground_truth==1)[0][0])
    # print(np.argmax(ground_truth))
    predicted = model.predict(np.expand_dims(image, 0))
    if np.argmax(ground_truth) != np.argmax(predicted):
        print("failed to correctly predict original image")

    results = [[] for x in range(num_classes)]
    for i in range(num_classes):
        if i == np.where(ground_truth == 1)[0][0]:
            continue
        for j in range(number_of_examples):
            pgaExample, pgaGeneration = cgGA.generate(populationSize=16, generation=10000, inputImage=image, model=model,
                                                                          target=i, IMAGEDIMENSION=32)
            results[i].append(pgaGeneration)

            imgTitle = "targetedResult_test" + str(image_index)
            pgaImage = np.expand_dims(pgaExample, 0)
            failedpred = model.predict(pgaImage)
            print(f'predictions: {class_labels[np.argmax(failedpred)]}')
            imgTitle = "./" + resultfolderpath + "/img/" + str(imgTitle) + str(j) + "_" + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(pgaExample, image, imgTitle)
            #
            # print(imgTitle)
    print_result(results)


def print_result(result):
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

    for i in range(num_classes):
        if len(result[i]) != 0:
            print(f'Average number of generation for {class_labels[i]} is: {sum(result[i])/len(result[i])}')


def main(args):
    # adversarialTraining("resnet20.h5", "adversarial_training", count=100)
    # load_and_train("resnet20.h5", "adversarial_training", count=100)
    calculate_distance("resnet20.h5", "calculate_distance", image_index=1001, number_of_examples=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
