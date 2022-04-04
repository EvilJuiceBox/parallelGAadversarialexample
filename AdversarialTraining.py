import argparse
import os
import time

import numpy as np
from keras.models import load_model

import AdvGenerator
import coursegrained_ga
import high_conf_cgga
import traditional_ga
import high_conf_ga
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
from ensembleMain import Result, saveResults


def adversarialTraining(filepath, resultfolderpath, skip=0, count=500):
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

    total = 0
    starting_index = 1000
    i = skip

    cgGA = coursegrained_ga.CourseGrainedGeneticAlgorithm()

    AT_X = []
    AT_Y = []

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        # DeepNeuralNetworkUtil.showImg(image)

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction:
            # generate adversarial example here.
            total += 1
            pgaStartTime = time.time()
            pgaExample, pgaGeneration = cgGA.generate(populationSize=16, generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            pgaEndtime = time.time() - pgaStartTime

            print(f"\tpgaModelTime: {pgaEndtime}\n")

            if pgaGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            # imgTitle = "cggaResult_test" + str(starting_index + i)
            pgaImage = np.expand_dims(pgaExample, 0)
            failedpred = model.predict(pgaImage)
            # imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
            #     DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            # DeepNeuralNetworkUtil.saveImg(pgaExample, image, imgTitle)
            #
            # print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, pgaExample, 32)
            l2n = AdvGenerator.compare(image, pgaExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {pgaGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')
            print(f'Time: {pgaEndtime}')

            AT_X.append(pgaExample)
            AT_Y.append(groundtruth)

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(AT_X, resultfolderpath, "adversarial_training_X")  # dumps the result, overwrites each old file.
            saveResults(AT_Y, resultfolderpath, "adversarial_training_Y")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


    #Adds 80% of the adversarial training set to the base training.
    x_train.append(AT_X[:count*0.8])
    y_train.append(AT_Y[:count*0.8])


    # train model
    print('Training...')
    # changing epoch from 100-200, batch size from 128 - 256
    #original paper" 256

    # 128, 200 epoch
    # loss: 1.438
    # accu: 0.799

    # 64, 125 epoch
    # Loss: 1.150
    # Accuracy: 0.824

    # standardised, 64, 125
    # Loss: 1.141
    # Accuracy: 0.827

    # 32, 200


    #Loss:     0.711
    # Accuracy: 0.884

    model.fit(x_train, y_train, x_test, y_test)

    #evaluate how much of the original is correct.
    model.evaluate(AT_X[count*0.2:], AT_Y[count*0.2:])


def load_and_train(filepath, resultfolderpath, count=500):
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

    import pickle
    pickle_in = open("./" + resultfolderpath + "/adversarial_training_X.pickle", "rb")
    AT_X = pickle.load(pickle_in)
    pickle_in = open("./" + resultfolderpath + "/adversarial_training_Y.pickle", "rb")
    AT_Y = pickle.load(pickle_in)

    # Adds 80% of the adversarial training set to the base training.
    # x_train.append(AT_X[:count * 0.8])
    # y_train.append(AT_Y[:count * 0.8])
    np.append(x_train, AT_X[:int(count * 0.8)])
    np.append(y_train, AT_Y[:int(count * 0.8)])

    # train model
    print('Training...')
    # changing epoch from 100-200, batch size from 128 - 256
    # original paper" 256

    # 128, 200 epoch
    # loss: 1.438
    # accu: 0.799

    # 64, 125 epoch
    # Loss: 1.150
    # Accuracy: 0.824

    # standardised, 64, 125
    # Loss: 1.141
    # Accuracy: 0.827

    # 32, 200

    # Loss:     0.711
    # Accuracy: 0.884

    model.fit(x_train, y_train, x_test, y_test)

    # evaluate how much of the original is correct.
    model.evaluate(AT_X[int(count * 0.2):], AT_Y[int(count * 0.2):])

def test_AT_model(filepath, resultfolderpath, count=500):
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

    import pickle
    pickle_in = open("./" + resultfolderpath + "/adversarial_training_X.pickle", "rb")
    AT_X = pickle.load(pickle_in)
    pickle_in = open("./" + resultfolderpath + "/adversarial_training_Y.pickle", "rb")
    AT_Y = pickle.load(pickle_in)
    model.self_evaluate(AT_X[int(count * 0.2):], AT_Y[int(count * 0.2):])

def main(args):
    # adversarialTraining("resnet20.h5", "adversarial_training", count=100)
    # load_and_train("resnet20.h5", "adversarial_training", count=100)
    test_AT_model("adversarial_training_dnn.h5", "adversarial_training", count=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
