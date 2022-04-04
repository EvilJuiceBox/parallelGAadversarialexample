import pickle
import argparse
import numpy as np
from ensembleTest import Result
from DeepNeuralNetwork import DeepNeuralNetworkUtil

def difference(x, y):
    diff = y - x
    return 100 * (diff / abs(x))


def main(args):
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

    pickle_in = open("./singleDNNover30image/regularmodelresults.pickle", "rb")
    collection = pickle.load(pickle_in)
    # print(type(collection))
    # print(collection)
    modelGen, modelL1, modelL2, modelTime = 0, 0, 0, 0
    time1 = 0
    count = len(collection)

    j = 0
    for modelResult in collection:
        j = j + 1

        # if j == 21:
        #     print(DeepNeuralNetworkUtil.getClassLabel(modelResult.label))
        #     pickle_out = open("./imagedump.pickle", "wb")
        #     pickle.dump(modelResult.image, pickle_out)
        #     pickle_out.close()
        #     exit(0)

        print(f"\nnumber{j}")
        if modelResult.generation >= 1000:
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
        print(f'Incorrect labels are: {class_labels[np.argmax(modelResult.label)]}')
        print(f'Generations took to generate model: {modelResult.generation}')
        print(f'L1 norm difference: {modelResult.l1norm}')
        print(f'L2 norm difference: {modelResult.l2norm}')
        print(f'Time difference: {modelResult.duration}')
        print(" ")

        time1 = time1 + modelResult.duration

        modelGen += modelResult.generation
        modelL1 += modelResult.l1norm
        modelL2 += modelResult.l2norm
        modelTime += modelResult.duration


    print(f"\nAverage Number of Generations for model - {modelGen/count}")
    print(f"Average Number of l1 for model - {modelL1/count}")
    print(f"Average Number of l2 for model - {modelL2/count}")
    print(f"Average Number of time for model - {modelTime/count}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
