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

    pickle_in = open("./traditional_ga/gaalgorithmresult.pickle", "rb")
    collection = pickle.load(pickle_in)
    # print(type(collection))
    # print(collection)

    # pickle_in = open("./high_conf_ga/highconfalgorithmresult.pickle", "rb")
    pickle_in = open("./parallel_ga/pgaalgorithmresult.pickle", "rb")
    pgaCollection = pickle.load(pickle_in)
    count = len(collection)

    genCount, l1Count, l2Count = 0, 0, 0
    gen, l1norm, l2norm = 0, 0, 0
    pgaGen, pgaL1, pgaL2, pgaTime = 0, 0, 0, 0
    modelGen, modelL1, modelL2, modelTime = 0, 0, 0, 0

    time1, time2 = 0, 0

    j = 0
    for modelResult, pgaResult in zip(collection, pgaCollection):
        j = j + 1

        # if j == 21:
        #     print(DeepNeuralNetworkUtil.getClassLabel(modelResult.label))
        #     pickle_out = open("./imagedump.pickle", "wb")
        #     pickle.dump(modelResult.image, pickle_out)
        #     pickle_out.close()
        #     exit(0)

        print(f"\nnumber{j}")
        if modelResult.generation == -1 or pgaResult.generation == -1:
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
        print(f'Incorrect labels are: {class_labels[np.argmax(modelResult.label)]} and {class_labels[np.argmax(pgaResult.label)]}')
        print(f'Generations took to generate model: {modelResult.generation} and {pgaResult.generation}')
        print(f'L1 norm difference: {modelResult.l1norm} and {pgaResult.l1norm}')
        print(f'L2 norm difference: {modelResult.l2norm} and {pgaResult.l2norm}')
        print(f'Time difference: {modelResult.duration} and {pgaResult.duration}')
        print(" ")

        time1 = time1 + modelResult.duration
        time2 = time2 + pgaResult.duration

        pgaGen += pgaResult.generation
        pgaL1 += pgaResult.l1norm
        pgaL2 += pgaResult.l2norm
        pgaTime += pgaResult.duration

        modelGen += modelResult.generation
        modelL1 += modelResult.l1norm
        modelL2 += modelResult.l2norm
        modelTime += modelResult.duration

        gen += pgaResult.generation - modelResult.generation
        if modelResult.generation < pgaResult.generation:
            genCount += 1

        l1norm += pgaResult.l1norm - modelResult.l1norm
        if modelResult.l1norm < pgaResult.l1norm:
            l1Count += 1

        l2norm += pgaResult.l2norm - modelResult.l2norm
        if modelResult.l2norm < pgaResult.l2norm:
            l2Count += 1

    print("--------------Summary--------------")
    print(f"\nNumber of Generation difference on average: {gen/count}")
    print(f"Number of generation higher for ensemble: {genCount}")
    print(f"\nNumber of l1 difference on average: {l1norm / count}")
    print(f"Number of l1 higher for ensemble: {l1Count}")
    print(f"\nNumber of l2 difference on average: {l2norm/count}")
    print(f"Number of l2 higher for ensemble: {l2Count}")

    print(f"\nAverage Number of Generations for model - {modelGen/count} and ensemble - {pgaGen/count}. It is {difference(modelGen/count, pgaGen/count)} percent higher.")
    print(f"Average Number of l1 for model - {modelL1/count} and ensemble - {pgaL1/count}. It is {difference(modelL1/count, pgaL1/count)} percent higher.")
    print(f"Average Number of l2 for model - {modelL2/count} and ensemble - {pgaL2/count}. It is {difference(modelL2/count, pgaL2/count)} percent higher.")
    print(f"Average Number of time for model - {modelTime/count} and ensemble - {pgaTime/count}. It is {difference(modelTime/count, pgaTime/count)} percent higher.")

    print(f"\nAverage time duration for each img (model vs ensemble): {time1/count} : {time2/count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
