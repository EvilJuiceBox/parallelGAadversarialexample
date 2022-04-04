import argparse
import os
import time

import numpy as np
from keras.models import load_model
from PIL import Image

import AdvGenerator
import targetedGeneticAlgorithm
import coursegrained_ga
import high_conf_cgga
import parallel_ga
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
from ensembleMain import Result, saveResults


def calculate_perturbation(filepath, resultfolderpath, image_index=0):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # # suppress tensorflow error output
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # import tensorflow as tf
    # tf.logging.set_verbosity(tf.logging.ERROR)

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
    # cgGA = coursegrained_ga.CourseGrainedGeneticAlgorithm()
    # hgGA = high_conf_cgga.HighConfidenceGeneticAlgorithm()
    # tGA = targetedGeneticAlgorithm.TargetedGeneticAlgorithm()
    pGA = parallel_ga.ParallelAlgorithm()

    image = x_test[image_index]
    ground_truth = y_test[image_index]
    #
    # print(ground_truth)
    # print(np.where(ground_truth==1)[0][0])
    # print(np.argmax(ground_truth))
    predicted = model.predict(np.expand_dims(image, 0))
    print(f"The original predicted label is {DeepNeuralNetworkUtil.getClassLabel(np.argmax(predicted))} with a confidence of {predicted[0][np.argmax(predicted)]}")
    if np.argmax(ground_truth) != np.argmax(predicted):
        print("failed to correctly predict original image")


    print(f"max entry: {np.amax(image)}")
    print(f"min entry: {np.amin(image)}")
    # advImage, generationCount = cgGA.generate(populationSize=16, generation=10000,
    #                                           inputImage=image, model=model,
    #                                           y_truth=np.argmax(ground_truth), IMAGEDIMENSION=32)

    advImage, generationCount = pGA.generate(populationSize=16, generation=10000,
                                              inputImage=image, model=model,
                                              y_truth=np.argmax(ground_truth), IMAGEDIMENSION=32)

    # advImage, generationCount = tGA.generate(populationSize=16, generation=10000,
    #                                           inputImage=image, model=model,
    #                                           target=4, IMAGEDIMENSION=32)

    if generationCount == -1:
        print("Failed to generate adversarial example")

    DeepNeuralNetworkUtil.saveImg(image, image, "./" + resultfolderpath + "/" + "original" + str(DeepNeuralNetworkUtil.getClassLabel(ground_truth)) + ".png")
    failedpred = model.predict(np.expand_dims(advImage, 0))
    imgTitle = "./" + resultfolderpath + "/" + "generatedExample" + str(image_index) + str(
        DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
    DeepNeuralNetworkUtil.saveImg(advImage, image, imgTitle)
    print(failedpred)
    print(
        f"The resulting predicted label is {DeepNeuralNetworkUtil.getClassLabel(failedpred)} with a confidence of {failedpred[0][np.argmax(failedpred)]}")


    perturbation_layer = DeepNeuralNetworkUtil.inverseStandardisation(advImage) - DeepNeuralNetworkUtil.inverseStandardisation(image)
    perturbation_layer = ((perturbation_layer / 256) * 128) + 128

    img = Image.fromarray(np.uint8(perturbation_layer.reshape(input_shape)), 'RGB').resize(
        (32, 32))
    img.save("./" + resultfolderpath + "/" + "perturbation_noise_layer.png")
    print(imgTitle)


def main(args):
    calculate_perturbation("resnet20.h5", "pga_test", image_index=1001)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
