from train_90acc_ensemble import getEnsemble
from keras.models import load_model
import numpy as np
import AdvGenerator
from PIL import Image
import argparse
import os
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil, Ensemble
import time
from ensembleMain import Result, saveResults
import coursegrained_ga, traditional_ga, high_conf_cgga
from scipy.ndimage.filters import gaussian_filter


"""
generates adversarial examples with the traditional GA approach
filepath: filename of the model to be used, assumed to be kumar model
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateResults(filepath, resultfolderpath, skip=0, count=30):
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
    cgGA = coursegrained_ga.CourseGrainedGeneticAlgorithm()
    # cgGA = high_conf_cgga.HighConfidenceGeneticAlgorithm()
    numberOfCorrected = 0

    while total < count:
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
            print(f"correct prediction is: {DeepNeuralNetworkUtil.getClassLabelByIndex(groundtruth)}")
            total += 1
            pgaStartTime = time.time()
            pgaExample, pgaGeneration = cgGA.generate(populationSize=7, generation=1000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            pgaEndtime = time.time() - pgaStartTime

            print(f"\tpgaModelTime: {pgaEndtime}\n")

            if pgaGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue


            # failedpred = model.predict(pgaImage)
            failedpred = DeepNeuralNetworkUtil.predict(model, pgaExample)
            # DeepNeuralNetworkUtil.show(pgaExample, title="advex")
            # imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
            #     DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            # DeepNeuralNetworkUtil.saveImg(pgaExample, image, imgTitle)

            print("applying singular vector decomposition")
            # DeepNeuralNetworkUtil.saveImg(pgaExample, DeepNeuralNetworkUtil.standarisation(target), "SVD test/pgaImage.png", 32)
            # reconstructedImg = gaussian_filter(pgaExample, sigma=0.5)
            reconstructedImg = preprocess(pgaExample, k=10, beta=0.1)
            # reconstructedImg = np.expand_dims(reconstructedImg, 0)
            # svdpredict = model.predict(reconstructedImg)
            svdpredict = DeepNeuralNetworkUtil.predict(model, reconstructedImg)
            # DeepNeuralNetworkUtil.show(reconstructedImg, title="svd")
            if groundtruth == np.argmax(svdpredict):
                numberOfCorrected += 1

            print(" ----- end of image test ----- \n\n")

    # count the toal
    print(f"Number of examples corrected by SVD: {numberOfCorrected}")
    print(f"Total number of adversarial examples generated: {count}")


def testSampleData(filepath, skip=0, count=1000):
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
    cgGA = coursegrained_ga.CourseGrainedGeneticAlgorithm()
    numberOfCorrected = 0

    while total < count:
        i += 1
        print(i)

        image = x_test[starting_index + i]

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction:
            print("applying gaussian blur algorithm")
            # DeepNeuralNetworkUtil.saveImg(pgaExample, DeepNeuralNetworkUtil.standarisation(target), "SVD test/pgaImage.png", 32)

            # image = DeepNeuralNetworkUtil.inverseStandardisation(image)
            reconstructedImg = preprocess(image, 10, 0.05)
            print(f"Euclidean distance between two imageS: {cgGA.compare(reconstructedImg, image, 32)}")

            # reconstructedImg = gaussian_filter(image, sigma=0.5)
            # reconstructedImg = DeepNeuralNetworkUtil.standarisation(reconstructedImg)
            reconstructedImg = np.expand_dims(reconstructedImg, 0)

            svdpredict = model.predict(reconstructedImg)
            # svdpredict = DeepNeuralNetworkUtil.predict(model, reconstructedImg)
            # DeepNeuralNetworkUtil.show(reconstructedImg, title="svd")

            if groundtruth == np.argmax(svdpredict):
                numberOfCorrected += 1
            total += 1
            print(" ----- end of image test ----- ")

    # count the toal
    print(f"Number of examples correctedly predicted: {numberOfCorrected}")
    print(f"Total number of adversarial examples generated: {count}")


def preprocess(image, k=10, beta=0.05):
    reconstructedImg = DeepNeuralNetworkUtil.evenImage(image, beta)
    return singularValueDecomposition(reconstructedImg, k)


def singularValueDecomposition(image, k):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    rU, rSigma, rV = np.linalg.svd(r)
    gU, gSigma, gV = np.linalg.svd(g)
    bU, bSigma, bV = np.linalg.svd(b)

    reconstr = rU[:, 0:k] @ np.diag(rSigma)[0:k, 0:k] @ rV[0:k, :]
    reconstg = gU[:, 0:k] @ np.diag(gSigma)[0:k, 0:k] @ gV[0:k, :]
    reconstb = bU[:, 0:k] @ np.diag(bSigma)[0:k, 0:k] @ bV[0:k, :]

    result = np.dstack((reconstr, reconstg, reconstb))
    # result = np.dstack((r, g, b))
    return result

def main(args):
    generateResults("kumardnn.h5", "SVD test", count=30)
    # testSampleData("kumardnn.h5", count=1000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
