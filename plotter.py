import pickle
import matplotlib.pyplot as plt
import argparse
from ensembleMain import Result
import numpy as np
import math


def removeNanEntries(minrange, maxrange, args):
    import copy
    temp = []
    for element in args:
        counter = minrange
        i = 0
        results_with_nan = []
        while counter < maxrange:
            if i < len(element):
                # print(counter)
                if counter == element[i].testnumber:
                    results_with_nan.append(element[i])
                    i += 1
                else:
                    results_with_nan.append(None)
            else:
                results_with_nan.append(None)
            counter += 1
        temp.append(results_with_nan)

    result = copy.deepcopy(temp)

    # for element in temp:
    #     print(type(element), end = "")
    #     print(" ")
    for i in range(len(result[0])):
        containNan = False
        for j in range(len(temp)):
            if temp[j][i] is None:
                containNan = True
        if containNan:  # if there is a nan in the specific index, remove all items from the same index
            for k in result:
                k[i] = None  #remove all i index from every list

    # for element in result:
    #     print("[", end="")
    #     for element2 in element:
    #         if element2 is None:
    #             print("1", end=", ")
    #         else:
    #             print("0", end=", ")
    #     print("]")
    # print(" ")
    return result


# def plot(*filepath):
def plot(**kwargs):
    labels = []
    results = []

    maxrange = -math.inf
    minrange = math.inf

    for key, value in kwargs.items():
        labels.append(key)
        pickle_in = open(value, "rb")
        temp = pickle.load(pickle_in)

        # print(temp[0].testnumber)
        # print(temp[-1].testnumber)
        # obtain first and last testnumber of the experiments
        if temp[0].testnumber < minrange:
            minrange = temp[0].testnumber
        if temp[-1].testnumber > maxrange:
            maxrange = temp[-1].testnumber

        results.append(temp)

    clean_input = removeNanEntries(minrange, maxrange, results)
    #loads all file in to results
    # for element in filepath:
    #     pickle_in = open(element, "rb")
    #     temp = pickle.load(pickle_in)
    #
    #     print(temp[0].testnumber)
    #     print(temp[-1].testnumber)
    #     # obtain first and last testnumber of the experiments
    #     if temp[0].testnumber < minrange:
    #         minrange = temp[0].testnumber
    #     if temp[-1].testnumber > maxrange:
    #         maxrange = temp[-1].testnumber
    #
    #     results.append(temp)


    x = range(minrange, maxrange)

    fig, ((genplt, l1plt), (l2plt, timeplt)) = plt.subplots(2, 2)

    for element, plotCount in zip(clean_input, range(len(results))):
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
        gen_mean = [np.nanmean(gen_temp)]*len(x)  # len(x) to make 30 mean in array for dimension matching
        genplt.plot(x, gen_mean, linestyle='--')
        genplt.scatter(x, generation, label=labels[plotCount], s=5)
        genplt.set_title("Generation Comparison")
        genplt.legend(bbox_to_anchor=(1.05, 1))
        genplt.set_xlabel("TestNumber")
        genplt.set_ylabel("Generation")

        l1n_temp = np.array(l1n, dtype=np.float64)
        l1_mean = [np.nanmean(l1n_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        print(f"Mean for {labels[plotCount]} l1n = {np.nanmean(l1n_temp)}")
        l1plt.scatter(x, l1n, label=labels[plotCount], s=5)
        l1plt.plot(x, l1_mean, linestyle='--')
        l1plt.set_title("L1 Comparison")
        # l1plt.legend(bbox_to_anchor=(1.05, 1))

        l2n_temp = np.array(l2n, dtype=np.float64)
        print(f"Mean for {labels[plotCount]} l2n = {np.nanmean(l2n_temp)}")
        l2_mean = [np.nanmean(l2n_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        l2plt.plot(x, l2_mean, linestyle='--')
        l2plt.scatter(x, l2n, label=labels[plotCount], s=5)
        l2plt.set_title("L2 Comparison")
        # l2plt.legend(bbox_to_anchor=(1.05, 1))

        time_temp = np.array(time, dtype=np.float64)
        print(f"Mean for {labels[plotCount]} time = {np.nanmean(time_temp)}")
        time_mean = [np.nanmean(time_temp)] * len(x)  # len(x) to make 30 mean in array for dimension matching
        timeplt.plot(x, time_mean, linestyle='--')
        timeplt.scatter(x, time, label=labels[plotCount], s=5)
        timeplt.set_title("Time Comparison")
        # timeplt.legend(bbox_to_anchor=(1.05, 1))

        print(" ")
    # naming the x axis
    # plt.xlabel('x - axis')
    # # naming the y axis
    # plt.ylabel('y - axis')
    #
    fig.tight_layout(pad=2.0)
    # function to show the plot
    plt.show()



def main(args):
    plot(homo3="./homogenous3_09012021/ensemblemodelresult.pickle",
         homo5n="./homogenous5nME_09012021/ensemblemodelresult.pickle",
         normal="./hetero3_09032021/regularmodelresults.pickle",  # regular model
         hetero3="./hetero3_09032021/ensemblemodelresult.pickle",
         homo5me="./homogenous5ME_09052021/ensemblemodelresult.pickle")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())