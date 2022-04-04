import pickle
import argparse
from ensembleTest import Result
from DeepNeuralNetwork import KumarDNN
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from PIL import Image

def main(args):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # load model
    file_path = 'ninetydnn.h5'
    print(f'Loading {file_path}...')
    resnet = load_model(file_path)
    # resnet.summary()
    model = KumarDNN(resnet)

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

    pickle_in = open("regularmodelresults.pickle", "rb")
    collection = pickle.load(pickle_in)
    # print(type(collection))
    # print(collection)

    pickle_in = open("ensemblemodelresult.pickle", "rb")
    ensembleCollection = pickle.load(pickle_in)
    count = len(collection)

    # modelResult = collection[2]
    # print(class_labels[np.argmax(model.predict(modelResult.image))])
    #
    #
    # input_shape = (32, 32, 3)
    # num_classes = 10
    # showimg = KumarDNN.inverseStandardisation(modelResult.image)
    # img = Image.fromarray(np.uint8(showimg.reshape(input_shape)), 'RGB').resize(
    #     (128, 128))
    # img.show()
    #
    #
    # print(class_labels[np.argmax(model.predict(KumarDNN.standarisation(np.expand_dims(x_test[2021], axis=0))))])


    genCount, l1Count, l2Count = 0, 0, 0
    gen, l1norm, l2norm = 0, 0, 0

    time1, time2 = 0, 0

    j = 0
    for modelResult, ensembleResult in zip(collection, ensembleCollection):
        j = j + 1
        print(f"\nnumber{j}")
        if modelResult.generation >= 1000 or ensembleResult.generation >= 1000 or j == 8:
            continue
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
        print(f'Time difference: {modelResult.duration} and {ensembleResult.duration}')
        print(" ")

        time1 = time1 + modelResult.duration
        time2 = time2 + ensembleResult.duration

        gen += ensembleResult.generation - modelResult.generation
        if modelResult.generation < ensembleResult.generation:
            genCount += 1


        l1norm += ensembleResult.l1norm - modelResult.l1norm
        if modelResult.l1norm < ensembleResult.l1norm:
            l1Count += 1

        l2norm += ensembleResult.l2norm - modelResult.l2norm
        if modelResult.l2norm < ensembleResult.l2norm:
            l2Count += 1


    print(f"\nNumber of Generation difference on average: {gen/count}")
    print(f"Number of generation higher for ensemble: {genCount}")
    print(f"\nNumber of l1 difference on average: {l1norm / count}")
    print(f"Number of l1 higher for ensemble: {l1Count}")
    print(f"\nNumber of l2 difference on average: {l2norm/count}")
    print(f"Number of l2 higher for ensemble: {l2Count}")

    print(f"\nAverage time duration for each img (model vs ensemble): {time1/count} : {time2/count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
