import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

    pickle_in = open("./50individualpga/pgaalgorithmresult.pickle", "rb")
    collection = pickle.load(pickle_in)
    # print(type(collection))
    # print(collection)

    # pickle_in = open("./high_conf_ga/highconfalgorithmresult.pickle", "rb")
    pickle_in = open("./50individualfgga_3grid/pgaalgorithmresult.pickle", "rb")
    pgaCollection = pickle.load(pickle_in)
    count = len(collection)

    genCount, l1Count, l2Count = 0, 0, 0
    gen, l1norm, l2norm = 0, 0, 0
    pgaGen, pgaL1, pgaL2, pgaTime = 0, 0, 0, 0
    modelGen, modelL1, modelL2, modelTime = 0, 0, 0, 0

    time1, time2 = 0, 0

    numprocessed = 0
    starting_index = 1000

    gaCollectionSynced = []
    pgaCollectionSynced = []
    x = []

    outlier_ga = []
    outlier_pga = []

    counter = 0
    while numprocessed < 30:
        modelResult = next((x for x in collection if x.testnumber == starting_index + counter), None)
        pgaResult = next((x for x in pgaCollection if x.testnumber == starting_index + counter), None)
        counter += 1
        if not [x for x in (modelResult, pgaResult) if x is None]:
            print(f"\nnumber{starting_index + counter - 1}")
            if modelResult.generation == -1 or pgaResult.generation == -1:
                continue

            # ignore outliers
            # if modelResult.generation >= 1500 or pgaResult.generation >= 1500:
            #     continue
            gaCollectionSynced.append(modelResult)
            pgaCollectionSynced.append(pgaResult)
            x.append(starting_index + counter)
            print(f"Test number: {modelResult.testnumber} and {pgaResult.testnumber}")
            print(
                f'Incorrect labels are: {class_labels[np.argmax(modelResult.label)]} and {class_labels[np.argmax(pgaResult.label)]}')
            print(f'Generations took to generate model: {modelResult.generation} and {pgaResult.generation}')
            print(f'L1 norm difference: {modelResult.l1norm} and {pgaResult.l1norm}')
            print(f'L2 norm difference: {modelResult.l2norm} and {pgaResult.l2norm}')
            print(f'Time difference: {modelResult.duration} and {pgaResult.duration}')
            print(" ")

            # add outliers
            if modelResult.generation >= 1200:
                outlier_ga.append(modelResult.testnumber)

            if pgaResult.generation >= 1200:
                outlier_pga.append(pgaResult.testnumber)

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
            if modelResult.generation > pgaResult.generation:
                genCount += 1

            l1norm += pgaResult.l1norm - modelResult.l1norm
            if modelResult.l1norm > pgaResult.l1norm:
                l1Count += 1

            l2norm += pgaResult.l2norm - modelResult.l2norm
            if modelResult.l2norm > pgaResult.l2norm:
                l2Count += 1

            numprocessed += 1


    print("--------------Summary--------------")
    print(f"\nNumber of Generation difference on average: {gen/count}")
    print(f"Number of generation higher for tradGA: {genCount}")
    print(f"\nNumber of l1 difference on average: {l1norm / count}")
    print(f"Number of l1 higher for tradGA: {l1Count}")
    print(f"\nNumber of l2 difference on average: {l2norm/count}")
    print(f"Number of l2 higher for tradGA: {l2Count}")

    print(f"\nAverage Number of Generations for tradGA: {modelGen/count} and cgGA: {pgaGen/count}. Difference is {difference(pgaGen/count, modelGen/count)} percent.") # difference(modelGen/count, pgaGen/count)
    print(f"Average Number of l1 for tradGA: {modelL1/count} and cgGA: {pgaL1/count}. Difference is {difference(pgaL1/count, modelL1/count)} percent.")
    print(f"Average Number of l2 for tradGA: {modelL2/count} and cgGA: {pgaL2/count}. Difference is {difference(pgaL2/count, modelL2/count)} percent.")
    print(f"Average Number of time for tradGA: {modelTime/count} and cgGA: {pgaTime/count}. Difference is {difference(pgaTime/count, modelTime/count)} percent.")

    print(f"\nAverage time duration for each img (tradGA vs cgGA): {time1/count} : {time2/count}")

    ## plotting
    fig, ((genplt, l1plt), (l2plt, timeplt)) = plt.subplots(2, 2, figsize=(6, 6))

    list_to_plot = [gaCollectionSynced, pgaCollectionSynced]
    labels = ["Traditional GA", "CgGA"]
    for element, plotCount in zip(list_to_plot, range(len(list_to_plot))):
        generation = []
        l1n = []
        l2n = []
        time = []

        for adversarialExample in element:
            if adversarialExample is not None:
                generation.append(adversarialExample.generation)
                l1n.append(adversarialExample.l1norm)
                l2n.append(adversarialExample.l2norm)
                time.append(adversarialExample.duration)
            if adversarialExample is None:
                generation.append(np.nan)
                l1n.append(np.nan)
                l2n.append(np.nan)
                time.append(np.nan)
        #     # testnum.append(adversarialExample.testnumber)

        # print(generation)
        # print("[", end="")
        # for temp in generation:
        #     print("{0:03.0f}".format(temp), end=", ")
        # print("]")
        #
        # print(f"x len: {len(x)}, ylen: {len(generation)}")

        gen_temp = np.array(generation, dtype=np.float64)
        print(f"Mean for {labels[plotCount]} generation = {np.nanmean(gen_temp)}")
        gen_mean = [np.nanmean(gen_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        genplt.plot(x, gen_mean, linestyle='--')
        genplt.scatter(x, generation, label=labels[plotCount], s=5)
        genplt.set_title("Generation Comparison")
        # genplt.legend(bbox_to_anchor=(1.05, 1))
        genplt.set_xlabel("Test Image Number")
        genplt.set_ylabel("Generation")

        l1n_temp = np.array(l1n, dtype=np.float64)
        l1_mean = [np.nanmean(l1n_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        print(f"Mean for {labels[plotCount]} l1n = {np.nanmean(l1n_temp)}")
        l1plt.scatter(x, l1n, label=labels[plotCount], s=5)
        l1plt.plot(x, l1_mean, linestyle='--')
        l1plt.set_title("L0 Comparison")
        l1plt.set_xlabel("Test Image Number")
        l1plt.set_ylabel("Number of Modified Pixels")
        # l1plt.legend(bbox_to_anchor=(1.05, 1))

        l2n_temp = np.array(l2n, dtype=np.float64)
        print(f"Mean for {labels[plotCount]} l2n = {np.nanmean(l2n_temp)}")
        l2_mean = [np.nanmean(l2n_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        l2plt.plot(x, l2_mean, linestyle='--')
        l2plt.scatter(x, l2n, label=labels[plotCount], s=5)
        l2plt.set_title("L2 Comparison")
        l2plt.set_xlabel("Test Image Number")
        l2plt.set_ylabel("L2 Norm Difference")
        # l2plt.legend(bbox_to_anchor=(1.05, 1))

        time_temp = np.array(time, dtype=np.float64)
        print(f"Mean for {labels[plotCount]} time = {np.nanmean(time_temp)}")
        time_mean = [np.nanmean(time_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        timeplt.plot(x, time_mean, linestyle='--')
        timeplt.scatter(x, time, label=labels[plotCount], s=5)
        timeplt.set_title("Time Comparison")
        timeplt.set_xlabel("Test Image Number")
        timeplt.set_ylabel("Time (sec)")
        # timeplt.legend(bbox_to_anchor=(1.05, 1))

        print(" ")

    print("Outliers are as follows")
    print(f"ga: {outlier_ga}")
    print(f"pga: {outlier_pga}")
    # naming the x axis
    # plt.xlabel('x - axis')
    # # naming the y axis
    # plt.ylabel('y - axis')
    #
    fig.tight_layout(pad=2.0)
    # function to show the plot
    plt.show()
    # genplt.show

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
