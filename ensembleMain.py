from train_90acc_ensemble import getEnsemble
from keras.models import load_model
import numpy as np
import AdvGenerator
from PIL import Image
import argparse
import os
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil, Ensemble
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


def getEnsembleFromFolder(ensembleSize, folderPath):
    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import load_model

    model = Ensemble()
    for i in range(ensembleSize):
        file_path = './models/' + folderPath + '/dnn_ensemble_target_' + str(i) + '.h5'
        print(f'Loading {file_path}...')
        submodel = load_model(file_path)
        model.append(submodel)

    return model


"""
generates adversarial examples with the single model.
filepath: filename of the model to be used, assumed to be kumar model
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateSingleModelResults(filepath, resultfolderpath, ensemblefolderpath, skip=0, count=30, ensembleSize=5):
    ensembleModel = getEnsembleFromFolder(ensembleSize, ensemblefolderpath)
    # load model
    print(f'Loading {filepath}...')
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
    i = skip
    collection = []

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))
        ensemblemodelprediction = np.argmax(ensembleModel.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        print(f'ensemblemodelprediction: {ensemblemodelprediction}')

        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction and groundtruth == ensemblemodelprediction:
            # generate adversarial example here.
            total += 1
            singleStartTime = time.time()
            singleExample, singleGeneration = AdvGenerator.parallelGA(populationSize=7, generation=1000,
                                                                      inputImage=image, model=model,
                                                                      y_truth=groundtruth, IMAGEDIMENSION=32)
            singleEndTime = time.time() - singleStartTime

            print(f"\n\tSingleModelTime: {singleEndTime}\n")

            if singleGeneration >= 1000:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "modelResult_test" + str(starting_index + i)
            resnetImage = np.expand_dims(singleExample, 0)
            failedpred = model.predict(resnetImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(singleExample, image, imgTitle)

            print(imgTitle)
            print(f'Ground truth: {class_labels[groundtruth]}')

            l1n = AdvGenerator.getl1normdiff(image, singleExample, 32)
            l2n = AdvGenerator.compare(image, singleExample, 32)
            print(f'fail prediction for resnet: {failedpred}')
            print(f'Generations took to generate model: {singleGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            collection.append(
                Result("KumarDNN", singleGeneration, l1n, l2n, resnetImage, failedpred, singleEndTime, image,
                       starting_index + i))
            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "regularmodelresults")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


"""
generates adversarial examples with the single model.
filepath: filename of the model to be used, assumed to be kumar model
ensemblefilepath: foldername of the ensembles (e.g., hetero3)
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateEnsembleModelResults(filepath, resultfolderpath, ensemblefolderpath, skip=0, count=30, ensembleSize=5):
    ensembleModel = getEnsembleFromFolder(ensembleSize, ensemblefolderpath)
    # load model
    print(f'Loading {filepath}...')
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
    i = skip
    ensembleCollection = []

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))
        ensemblemodelprediction = np.argmax(ensembleModel.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        print(f'ensemblemodelprediction: {ensemblemodelprediction}')

        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction and groundtruth == ensemblemodelprediction:
            # generate adversarial example here.
            total += 1
            ensembleStartTime = time.time()
            ensembleExample, ensembleGeneration = AdvGenerator.parallelGA(populationSize=7, generation=1000,
                                                                          inputImage=image, model=ensembleModel,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            ensembleEndtime = time.time() - ensembleStartTime

            print(f"\tEnsembleModelTime: {ensembleEndtime}\n")

            if ensembleGeneration >= 1000:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "ensembleResult_test" + str(starting_index + i)
            ensembleImage = np.expand_dims(ensembleExample, 0)
            failedpred = ensembleModel.predict(ensembleImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(ensembleExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, ensembleExample, 32)
            l2n = AdvGenerator.compare(image, ensembleExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {ensembleGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            ensembleCollection.append(
                Result("Ensemble", ensembleGeneration, l1n, l2n, ensembleImage, failedpred, ensembleEndtime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(ensembleCollection, resultfolderpath, "ensemblemodelresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")

"""
generates adversarial examples with the single model.
filepath: filename of the model to be used, assumed to be kumar model
ensemblefilepath: foldername of the ensembles (e.g., hetero3)
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def resumeEnsembleModel(filepath, resultfolderpath, ensemblefolderpath, skip, count=30, ensembleSize=5):
    ensembleModel = getEnsembleFromFolder(ensembleSize, ensemblefolderpath)
    # load model
    print(f'Loading {filepath}...')
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
    i = skip

    ensembleCollection = loadFromPickle(resultfolderpath)

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))
        ensemblemodelprediction = np.argmax(ensembleModel.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        print(f'ensemblemodelprediction: {ensemblemodelprediction}')

        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction and groundtruth == ensemblemodelprediction:
            # generate adversarial example here.
            total += 1
            ensembleStartTime = time.time()
            ensembleExample, ensembleGeneration = AdvGenerator.parallelGA(populationSize=7, generation=1000,
                                                                          inputImage=image, model=ensembleModel,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            ensembleEndtime = time.time() - ensembleStartTime

            print(f"\tEnsembleModelTime: {ensembleEndtime}\n")
            if ensembleGeneration >= 1000:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "ensembleResult_test" + str(starting_index + i)
            ensembleImage = np.expand_dims(ensembleExample, 0)
            failedpred = ensembleModel.predict(ensembleImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(ensembleExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, ensembleExample, 32)
            l2n = AdvGenerator.compare(image, ensembleExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {ensembleGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            ensembleCollection.append(
                Result("Ensemble", ensembleGeneration, l1n, l2n, ensembleImage, failedpred, ensembleEndtime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(ensembleCollection, resultfolderpath, "ensemblemodelresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


def loadFromPickle(resultfolderpath):
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
    import pickle
    pickle_in = open("./" + resultfolderpath + "/ensemblemodelresult.pickle", "rb")
    ensembleCollection = pickle.load(pickle_in)
    count = len(ensembleCollection)
    print(f"Resuming ensemble experiment, printing existing adversarial example information. There is a total of {count} items in collection")
    for element in ensembleCollection:
        print(
            f"\tAdversarial Example #{element.testnumber}: mispredicted label = {class_labels[np.argmax(element.label)]}, duration = {element.duration}")

    return ensembleCollection

def saveResults(collection, folderpath, filename):
    import pickle

    pickle_out = open("./" + folderpath + "/" + filename + ".pickle", "wb")
    pickle.dump(collection, pickle_out)
    pickle_out.close()



def main(args):
    # experiment to genrate DNN results with ensemble of 3 different architectures
    # generateEnsembleModelResults("kumardnn.h5", "homogenous3_09012021", "homogenous3", ensembleSize=3)
    generateEnsembleModelResults("kumardnn.h5", "homogenous5nME_09012021", "homogenous5nME", ensembleSize=5)  # 5 nonmutually exclusive items
    # generateEnsembleModelResults("kumardnn.h5", "homogenous5ME_09052021", "homogenous5ME", ensembleSize=5)  # 5 mutually exclusive items
    # loadFromPickle("singleDNN08222021")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
