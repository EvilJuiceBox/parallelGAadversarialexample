from Candidate import Candidate
from AdversarialExample import AdversarialExample
from operator import itemgetter, attrgetter
import numpy as np
import os
from DeepNeuralNetwork import Ensemble


def getEnsemble(ensembleSize):
    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import load_model

    model = Ensemble()
    for i in range(ensembleSize):
        file_path = 'models/hetero3/dnn_ensemble_target_' + str(i) + '.h5'
        print(f'Loading {file_path}...')
        submodel = load_model(file_path)
        model.append(submodel)

    return model


def main():
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
    model = getEnsemble(3)
    model.evaluate(x_test, y_test)
    model.recursive_evaluate(x_test, y_test)


if __name__ == '__main__':
    main()