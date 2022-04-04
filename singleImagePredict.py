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
import coursegrained_ga

def main(args):
    filepath = "kumardnn.h5"
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

    image = DeepNeuralNetworkUtil.loadImg("parallel_ga/gaResult_test1002truck.png")
    image = (image - mean) / (std + 1e-7)

    y = model.predict(image)

    print(f'Prediction: {y}')
    print(f'Prediction: {class_labels[np.argmax(y)]}')
    print(f'Prediction: {np.argmax(y)}')
    print(f'Prediction: {y[0][np.argmax(y)]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
