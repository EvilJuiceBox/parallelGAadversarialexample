from train_90acc_ensemble import getEnsemble
from keras.models import load_model
import numpy as np
import AdvGenerator
from PIL import Image
import argparse
import os
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
import time


class Result:
    def __init__(self, type, gen, l1, l2, img, mislabel, d, ori, num):
        self.model = type
        self.generation = gen
        self.l1norm = l1
        self.l2norm = l2
        self.image = img
        self.label = mislabel
        self.duration = d
        self.original = ori
        self.testnumber = num

    def print(self):
        print(f'fail prediction for resnet: {self.label}')
        print(f'Generations took to generate model: {self.generation}')
        print(f'L1 norm difference: {self.l1norm}')
        print(f'L2 norm difference: {self.l2norm}')


def compareprint(modelResult, ensembleResult):
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
    print(f'Incorrect labels are: {class_labels[np.argmax(modelResult.label)]} and {class_labels[np.argmax(ensembleResult.label)]}')
    print(f'Generations took to generate model: {modelResult.generation} and {ensembleResult.generation}')
    print(f'L1 norm difference: {modelResult.l1norm} and {ensembleResult.l1norm}')
    print(f'L2 norm difference: {modelResult.l2norm} and {ensembleResult.l2norm}')


def main(args):
    #define number of dnns

    # load model
    file_path = 'ninetydnn.h5'
    print(f'Loading {file_path}...')
    resnet = load_model(file_path)
    resnet.summary()
    model = KumarDNN(resnet)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.datasets import cifar10
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

    starting_index = 1000

    total = 0
    i = 0
    collection = []

    while total < 30:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')

        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction:
            # generate adversarial example here.
            total += 1
            resnetStartTime = time.time()
            resnetExample, resnetGeneration = AdvGenerator.parallelGA(populationSize=7, generation=1000, inputImage=image, model=model,
                                                             y_truth=groundtruth, IMAGEDIMENSION=32)
            resnetEndTime = time.time() - resnetStartTime

            imgTitle = "modelResult_test" + str(starting_index + i)
            resnetImage = np.expand_dims(resnetExample, 0)
            failedpred = model.predict(resnetImage)
            imgTitle = "./singleDNNover30image/" + str(imgTitle) + str(DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(resnetExample, image, imgTitle)

            print(imgTitle)
            print(f'Ground truth: {class_labels[groundtruth]}')

            l1n = AdvGenerator.getl1normdiff(image, resnetExample, 32)
            l2n = AdvGenerator.compare(image, resnetExample, 32)
            print(f'fail prediction for resnet: {failedpred}')
            print(f'Generations took to generate model: {resnetGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            collection.append(Result("KumarDNN", resnetGeneration, l1n, l2n, resnetImage, failedpred, resnetEndTime, image, starting_index + i))

    print("\n\n\n")

    import pickle
    pickle_out = open("./singleDNNover30image/regularmodelresults.pickle", "wb")
    pickle.dump(collection, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
