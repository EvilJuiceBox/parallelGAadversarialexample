from matplotlib import pyplot
from PIL import Image
from keras.datasets import cifar10
from DeepNeuralNetwork import KumarDNN
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
import argparse
from AdvGenerator import randomPixelMutate
from DeepNeuralNetwork import DeepNeuralNetworkUtil


def show_imgs(X):
    input_shape = (32, 32, 3)
    pyplot.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            pyplot.subplot2grid((4,4),(i,j))
            img = Image.fromarray(np.uint8(X[k].reshape(input_shape)), 'RGB').resize(
                (128, 128))
            pyplot.imshow(img)
            k = k+1
    # show the plot
    pyplot.show()


def main(args):
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

    x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # load model
    file_path = 'ninetydnn.h5'
    print(f'Loading {file_path}...')
    resnet = load_model(file_path)
    # resnet.summary()
    model = KumarDNN(resnet)

    image = DeepNeuralNetworkUtil.loadImg("./kiraGA/modelResult_test1002truck.png")
    image = image.astype('uint8')
    print(DeepNeuralNetworkUtil.getClassLabel(model.standardisePredict(image)))

    # import pickle
    # pickle_in = open("./imagedump.pickle", "rb")
    # testimg = pickle.load(pickle_in)
    #
    # print("from pickle")
    # print(DeepNeuralNetworkUtil.getClassLabel(model.predict(testimg)))

    # model.evaluate(x_test, y_test)
    # img = x_test[2020]
    # showimg = Image.fromarray(np.uint8(img.reshape(input_shape)), 'RGB').resize(
    #     (128, 128))
    # showimg.show()
    #
    # std_img = KumarDNN.standarisation(img)
    #
    # for i in range(150):
    #     randomPixelMutate(std_img, 16, 16)
    # reverted_img = KumarDNN.inverseStandardisation(std_img)
    #
    # showimg = Image.fromarray(np.uint8(reverted_img.reshape(input_shape)), 'RGB').resize(
    #     (128, 128))
    # showimg.show()
    #
    # print(class_labels[np.argmax(model.standardisePredict(np.expand_dims(img, 0)))])


    #
    # # mean-std normalization
    # # mean = np.mean(x_train, axis=(0, 1, 2, 3))
    # # std = np.std(x_train, axis=(0, 1, 2, 3))
    # # x_train = (x_train - mean) / (std + 1e-7)
    # # x_test = (x_test - mean) / (std + 1e-7)
    # #
    #
    # labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # indices = []
    # for i in range(16):
    #     indices = np.argmax(model.predict(np.expand_dims(x_test[i], 0)))
    #     print(str(i) + labels[indices])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
