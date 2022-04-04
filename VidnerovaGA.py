# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import math
from PIL import Image
import random
import copy
from Candidate import Candidate
from AdversarialExample import AdversarialExample
from operator import itemgetter, attrgetter
from DeepNeuralNetwork import DifferentEnsemble
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
import time
from keras.models import load_model
from ensembleTest import Result

# takes in an image and modifies the pixels located at x y to new rgb value
def modifyImageByPixel(image, x, y, newvalue):
    image[x][y] = newvalue


def mutate1(candidate, IMAGEDIMENSION):
    target = candidate.getImage()
    for i in range(3):  # modify 3 random pixel, number is selected arbitrary for now
        randomPixelMutate(target, random.randint(0, IMAGEDIMENSION - 1), random.randint(0, IMAGEDIMENSION - 1))
    candidate.setImage(target)


def mutate(candidate, IMAGEDIMENSION):
    target = candidate.getImage()
    randomPixelMutate(target, random.randint(0, IMAGEDIMENSION - 1), random.randint(0, IMAGEDIMENSION - 1))
    candidate.setImage(target)


#modified to use standardised cifar10 values
def randomPixelMutate(image, x, y):
    channel = random.randint(0, 2)  # find a random channel to modify

    #commented out 255 due to stdisation of second dnn.
    value = image[x][y][channel]  # * 255  # rgb channel value from 0-255
    # value = (value * (64.1500758911213 + 1e-7) + 120.70756512369792)
    # return value to 0-255 (unstandardisation)

    # print("Original:" + str(value))
    # print(f"255*0.05 stadnardised is : {DifferentEnsemble.standardisation(255*0.05)}")
    # print(f"255 stadnardised is : {DifferentEnsemble.standardisation(255)}")
    # print(f"0 stadnardised is : {DifferentEnsemble.standardisation(0)}")
    # print(f"Difference: {(DifferentEnsemble.standardisation(255) - DifferentEnsemble.standardisation(0))*0.05}")

    if random.randint(0, 1) == 0:
        value = value + (0.19875268739766835)  # increase pixel value by 10 percent
        if value > 2.093410378732659:
            value = 2.093410378732659
    else:
        value = value - (0.19875268739766835)  # decrease value by 10 percent
        if value < -1.8816433692207077:
            value = -1.8816433692207077

    # restandardisation, avoid big calculations during predict.
    # value = (value - 120.70756512369792) / (64.1500758911213 + 1e-7)
    image[x][y][channel] = value  # / 255

    # print("New:" + str(image[x][y][channel] * 255))
    # print(" ")
    
    
# def randomPixelMutate(image, x, y):
#     channel = random.randint(0, 2)  # find a random channel to modify
#
#     #commented out 255 due to stdisation of second dnn.
#     value = image[x][y][channel]  # * 255  # rgb channel value from 0-255
#     value = (value * (64.1500758911213 + 1e-7) + 120.70756512369792)
#     # return value to 0-255 (unstandardisation)
#
#     # print("Original:" + str(value))
#     print(f"255 stadnardised is : {DifferentEnsemble.standardisation(255*0.05)}")
#
#     if random.randint(0, 1) == 0:
#         value = value + (255 * 0.05)  # increase pixel value by 10 percent
#         if value > 255:
#             value = 255
#     else:
#         value = value - (255 * 0.05)  # decrease value by 10 percent
#         if value < 0:
#             value = 0
#
#     # restandardisation, avoid big calculations during predict.
#     value = (value - 120.70756512369792) / (64.1500758911213 + 1e-7)
#     image[x][y][channel] = value  #/ 255
#
#     # print("New:" + str(image[x][y][channel] * 255))
#     # print(" ")

# crossover function that swaps the pixel values of two images past random x and y point.
def crossover(item1, item2, imageDimension):
    crossover_pointx = random.randint(0, imageDimension)
    crossover_pointy = random.randint(0, imageDimension)
    # print(str(crossover_pointx) + " :: " + str(crossover_pointy))
    temp1 = item1.getImage()
    temp2 = item2.getImage()

    # print("temp1:" + str(type(temp1)))
    # print("temp2: " + str(type(temp2)))

    for i in range(crossover_pointx):
        for j in range(crossover_pointy):
            tempPixel = copy.deepcopy(temp1[i][j])
            temp1[i][j] = temp2[i][j]
            temp2[i][j] = tempPixel
            # temp1[i][j], temp2[i][j] = temp2[i][j], temp1[i][j]

    item1.setImage(temp1)
    item2.setImage(temp2)

    return Candidate(copy.deepcopy(temp1)), Candidate(copy.deepcopy(temp2))


def survivorSelection(inputPopulation, cullsize, elitism, model, y):
    population = copy.deepcopy(inputPopulation)
    best = population[0:elitism]  # the best n performing member, saved for elitism

    temp = best
    for i in range(cullsize - elitism):  # pick the remaining survivors
        survivor, survivor1 = tournamentSelection(population, 3, model, y)
        temp.append(survivor)
        population = removeIndividual(population, survivor)  # prevent same individual from appearing twice

    temp = calculatePopulationFitness(temp, model, y)
    temp.sort(key=attrgetter("fitness"), reverse=False)
    return temp


def removeIndividual(iterable, item):
    for i, o in enumerate(iterable):
        if (o.getImage() == item.getImage()).all():
            del iterable[i]
            break
    return iterable


def parallelGAhighConf(populationSize, generation, inputImage, model, y_truth, IMAGEDIMENSION):
    print("parallelGA High Confidence start")
    islandCount = 3  # 3 island, might allow dynamic allocation later
    numberOfChildren = 3  # 3 children per generation.
    tournamentSize = 3  # tournament size
    islands = []
    for i in range(islandCount):
        population = []
        for j in range(populationSize):
            population.append(Candidate(inputImage))
        islands.append(population)

    for i in range(generation):
        for z in range(islandCount):
            # print("BEGINNING OF GENERATION: " + str(i))
            # generate children
            tempPopulation = []
            mutants = copy.deepcopy(islands[z])

            # Force each memeber of the population to mutate at least once.
            # copiedTemp = []
            # for j in range(len(tempPopulation)):
            #     copiedTemp.append(mutate(tempPopulation[j], IMAGEDIMENSION))
            #
            # tempPopulation = copiedTemp
            copytemp = copy.deepcopy(islands[z])
            for x in range(len(copytemp)):
                target = mutants[x]
                # mutate(target, IMAGEDIMENSION)
                tempPopulation.append(Candidate(target.getImage()))

            for j in range(numberOfChildren):
                offspring1, offspring2 = tournamentSelection(islands[z], tournamentSize, model, y_truth)
                # Crossover operation
                if random.random() < 0.6:
                    child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)
                else:
                    child1, child2 = offspring1, offspring2

                if random.random() < 0.1:
                    # Mutate operation and add to temp pop
                    mutate(child1, IMAGEDIMENSION)
                    mutate(child2, IMAGEDIMENSION)

                tempPopulation.append(Candidate(child1.getImage()))
                tempPopulation.append(Candidate(child2.getImage()))

            # cull population down to original size, and proceed to next gen.
            tempPopulation = calculatePopulationHighConfFitness(tempPopulation, model, y_truth)
            tempPopulation.sort(key=attrgetter("fitness"), reverse=False)

            if tempPopulation[0].getFitness() == -1:
                print("The solution was found at generation: " + str(i))
                return tempPopulation[0].getImage(), i

            islands[z] = survivorSelection(tempPopulation, populationSize, 3, model,
                                           y_truth)  # elitism of 3 per round, chosen arbitrary

        if i % 10 == 0:  # every 10 generation, migrate
            migrate = 0
            while migrate < populationSize:
                temp = islands[0][migrate]
                islands[0][migrate] = islands[1][migrate]
                islands[1][migrate] = islands[2][migrate]
                islands[2][migrate] = temp
                migrate += 1

        if i % 100 == 0:
            print("End of generation: " + str(i) + "; Best performing member: " + str(
                islands[0][0].getFitness()) + "; Worse performing member: " + str(
                islands[len(islands) - 1][0].getFitness()))
        # print("END OF GENERATION: " + str(i))

    return getBestMember(islands).getImage(), 9999


def parallelGA(populationSize, generation, inputImage, model, y_truth, IMAGEDIMENSION):
    print("parallelGA start")
    islandCount = 3  # 3 island, might allow dynamic allocation later
    numberOfChildren = 3  # 3 children per generation.
    tournamentSize = 3  # tournament size
    islands = []
    for i in range(islandCount):
        population = []
        for j in range(populationSize):
            population.append(Candidate(inputImage))
        islands.append(population)

    for i in range(generation):
        for z in range(islandCount):
            # print("BEGINNING OF GENERATION: " + str(i))
            # generate children
            tempPopulation = []
            mutants = copy.deepcopy(islands[z])

            # Force each memeber of the population to mutate at least once.
            # copiedTemp = []
            # for j in range(len(tempPopulation)):
            #     copiedTemp.append(mutate(tempPopulation[j], IMAGEDIMENSION))
            #
            # tempPopulation = copiedTemp
            copytemp = copy.deepcopy(islands[z])
            for x in range(len(copytemp)):
                target = mutants[x]
                # mutate(target, IMAGEDIMENSION)
                tempPopulation.append(Candidate(target.getImage()))

            for j in range(numberOfChildren):
                offspring1, offspring2 = tournamentSelection(islands[z], tournamentSize, model, y_truth)
                # Crossover operation
                if random.random() < 0.6:
                    child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)
                else:
                    child1, child2 = offspring1, offspring2

                if random.random() < 0.1:
                    # Mutate operation and add to temp pop
                    mutate(child1, IMAGEDIMENSION)
                    mutate(child2, IMAGEDIMENSION)

                tempPopulation.append(Candidate(child1.getImage()))
                tempPopulation.append(Candidate(child2.getImage()))

            # cull population down to original size, and proceed to next gen.
            tempPopulation = calculatePopulationFitness(tempPopulation, model, y_truth)
            tempPopulation.sort(key=attrgetter("fitness"), reverse=False)

            if tempPopulation[0].getFitness() == -1:
                print("The solution was found at generation: " + str(i))
                return tempPopulation[0].getImage(), i

            islands[z] = survivorSelection(tempPopulation, populationSize, 3, model,
                                           y_truth)  # elitism of 3 per round, chosen arbitrary

        if i % 10 == 0:  # every 10 generation, migrate
            migrate = 0
            while migrate < populationSize:
                temp = islands[0][migrate]
                islands[0][migrate] = islands[1][migrate]
                islands[1][migrate] = islands[2][migrate]
                islands[2][migrate] = temp
                migrate += 1

        if i % 100 == 0:
            print("End of generation: " + str(i) + "; Best performing member: " + str(
                islands[0][0].getFitness()) + "; Worse performing member: " + str(
                islands[len(islands) - 1][0].getFitness()))
        # print("END OF GENERATION: " + str(i))

    return getBestMember(islands).getImage(), 9999


def getBestMember(islands):
    best = islands[0][0]  # default first item of first islands
    for i in range(len(islands)):
        if best.getFitness() > islands[i][0].getFitness():
            best = islands[i][0]
    return best


# generates an image, populationSize, generation, and IMAGEDIMENSION are numbers
# inputImage is an inputImage from the CIFAR10 data
# model is a precompiled model
# y_truth is np.argmax(y), or y_test
def generateImageGA(populationSize, generation, inputImage, model, y_truth, IMAGEDIMENSION):
    population = []
    numberOfChildren = 3  # 3 children per generation.
    tournamentSize = 3  # tournament size
    # init population
    for i in range(populationSize):
        population.append(Candidate(inputImage))

    # while generation < 10:
    #     print("Generation: " + str(generation))
    #     # save(population, generation)
    #     generation += 1
    #     # fitness_evals += 100#each generation performs 100 fitness evaluations
    #
    #     # cross over population
    #     population = crossoverPopulation(population)
    #
    #     # mutate population
    #     for element in population:
    #         if random.random() < 0.8:
    #             mutate(element)
    for i in range(generation):
        # print("BEGINNING OF GENERATION: " + str(i))
        # generate children
        tempPopulation = []
        mutants = copy.deepcopy(population)

        # Force each memeber of the population to mutate at least once.
        # copiedTemp = []
        # for j in range(len(tempPopulation)):
        #     copiedTemp.append(mutate(tempPopulation[j], IMAGEDIMENSION))
        #
        # tempPopulation = copiedTemp
        copytemp = copy.deepcopy(population)
        for x in range(len(copytemp)):
            target = mutants[x]
            # mutate(target, IMAGEDIMENSION)
            tempPopulation.append(Candidate(target.getImage()))

        for j in range(numberOfChildren):
            offspring1, offspring2 = tournamentSelection(population, tournamentSize, model, y_truth)
            # Crossover operation
            if random.random() < 0.6:
                child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)
            else:
                child1, child2 = offspring1, offspring2

            if random.random() < 0.1:
            # Mutate operation and add to temp pop
                mutate(child1, IMAGEDIMENSION)
                mutate(child2, IMAGEDIMENSION)

            tempPopulation.append(Candidate(child1.getImage()))
            tempPopulation.append(Candidate(child2.getImage()))

        # cull population down to original size, and proceed to next gen.
        tempPopulation = calculatePopulationFitness(tempPopulation, model, y_truth)
        tempPopulation.sort(key=attrgetter("fitness"), reverse=False)

        if tempPopulation[0].getFitness() == -1:
            print(str(i))
            return tempPopulation[0].getImage(), i

        population = survivorSelection(tempPopulation, populationSize, 3, model,
                                       y_truth)  # elitism of 3 per round, chosen arbitrary
        if i % 100 == 0:
            print("End of generation: " + str(i) + "; Best performing member: " + str(
                population[0].getFitness()) + "; Worse performing member: " + str(
                population[len(population) - 1].getFitness()))
        # print("END OF GENERATION: " + str(i))
    return population[0].getImage(), 9999


# input image, tournament size (usually 3), returns 2 individuals that performs the best within the tournament selection
def tournamentSelection(inp, tournamentSize, model,
                        truth):  # returns 2 individuals via tournament selection (cannot be same)
    individuals = copy.deepcopy(inp)

    # for element in individuals:
    #     print(element)

    individuals = calculatePopulationFitness(individuals, model, truth)
    round1 = random.sample(individuals, tournamentSize)

    result1 = round1[0]
    for i in range(len(round1)):
        if round1[i].getFitness() < result1.getFitness():  # lower confidence is better
            result1 = round1[i]  # return the greatest individual from round1 sample

    individuals.remove(result1)  # remove round 1 individual

    round2 = random.sample(individuals, tournamentSize)

    result2 = round2[0]
    for i in range(len(round2)):
        if round2[i].getFitness() < result2.getFitness():
            result2 = round2[i]  # return the greatest individual from round1 sample

    return result1, result2


#   returns -1 if the image does not match ground truth
#   returns the confidence interval of the image otherwise
def getFitness(image, model, truth):
    # image = candidateInput.getImage()
    x = np.expand_dims(image, 0)
    y = model.predict(x)

    if truth != np.argmax(y):
        # if truth != np.argmax(y) and y[0][np.argmax(y)] > 0.9:  # and argmax is greater than 90% confidence
        return -1
    # returns the confidence level of the corresponding item
    # print("fitness estimate:")
    # print(y[0][np.argmax(y)])
    return y[0][np.argmax(y)]


def getFitnessOnConstraint(original, image, model, truth, IMAGEDIMENSION):
    x = np.expand_dims(image, 0)
    y = model.predict(x)
    lambda1 = 1
    lambda2 = 0.1

    if truth != np.argmax(y):
        # if truth != np.argmax(y) and y[0][np.argmax(y)] > 0.9:  # and argmax is greater than 90% confidence
        return -1

    # Base + L1 + L2
    # multiply base by 100 so the range is 0-100
    # L1 norm is number of pixel changes, range: (0, 32)
    # L2 norm is the sum of difference of pixel value squared
    fitness = 100 * y[0][np.argmax(y)] + lambda1 * getl1normdiff(original, image, IMAGEDIMENSION) + lambda2 * compare(original, image, IMAGEDIMENSION)


#   returns -1 if the image does not match ground truth
#   returns the confidence interval of the image otherwise
def getHighConfFitness(image, model, truth):
    # image = candidateInput.getImage()
    x = np.expand_dims(image, 0)
    y = model.predict(x)

    # if truth != np.argmax(y):
    if truth != np.argmax(y) and y[0][np.argmax(y)] > 0.9:  # and argmax is greater than 90% confidence
        return -1
    # returns the confidence level of the corresponding item
    # print("fitness estimate:")
    # print(y[0][np.argmax(y)])
    return y[0][np.argmax(y)]


def calculatePopulationHighConfFitness(population, model, truth):
    temp = []
    for sample in population:
        item = Candidate(sample.getImage())
        item.setFitness(getHighConfFitness(sample.getImage(), model, truth))
        temp.append(item)
    return temp


def calculatePopulationFitness(population, model, truth):
    temp = []
    for sample in population:
        item = Candidate(sample.getImage())
        item.setFitness(getFitness(sample.getImage(), model, truth))
        temp.append(item)
    return temp


def init(args):  # loads the pretrained model
    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import load_model

    file_path = 'cifar10_dnn.h5'
    print(f'Loading {file_path}...')
    model = load_model(file_path)
    model.summary()

    return model


# calculates the difference between the two images, average pixel value difference.
def compare(image1, image2, IMAGEDIMENSION):
    mse = 0
    for i in range(IMAGEDIMENSION):
        for j in range(IMAGEDIMENSION):
            for k in range(3):  # RGB channels
                mse += (int(image1[i][j][k]) - int(image2[i][j][k])) ** 2  # divided by 255 for new dnn

    return math.sqrt(mse)


# returns the number of changed pixels
def getl1normdiff(original, perturbed, IMAGEDIMENSION):
    result = 0
    for i in range(IMAGEDIMENSION):
        for j in range(IMAGEDIMENSION):
            gate = False
            for k in range(3):  # RGB channels
                if original[i][j][k] != perturbed[i][j][k]:
                    gate = True

            if gate:
                result += 1
    return result


def showImg(imgToDisplay):
    input_shape = (32, 32, 3)
    img = Image.fromarray(np.uint8(imgToDisplay.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    img.show()


def getDifference(original, perturbed, IMAGEDIMENSION):
    result = copy.deepcopy(original)
    for i in range(IMAGEDIMENSION):
        for j in range(IMAGEDIMENSION):
            for k in range(3):  # RGB channels
                result[i][j][k] -= perturbed[i][j][k]
    return result


# main script
def main(model):
    # load model
    file_path = 'ninetydnn.h5'
    print(f'Loading {file_path}...')
    resnet = load_model(file_path)
    resnet.summary()
    model = KumarDNN(resnet)

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
    i = 0
    collection = []


    modelGen, modelL1, modelL2, totaltime = 0, 0, 0, 0

    while total < 1:
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
            total += 1
            resnetStartTime = time.time()
            resnetExample, resnetGeneration = generateImageGA(populationSize=50, generation=10000,
                                                                      inputImage=image, model=model,
                                                                      y_truth=groundtruth, IMAGEDIMENSION=32)
            resnetEndTime = time.time() - resnetStartTime

            imgTitle = "modelResult_test" + str(starting_index + i)
            resnetImage = np.expand_dims(resnetExample, 0)
            failedpred = model.predict(resnetImage)
            imgTitle = "./vidnerovaGA/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(resnetExample, image, imgTitle)

            print(imgTitle)
            print(f'Ground truth: {class_labels[groundtruth]}')

            l1n = getl1normdiff(image, resnetExample, 32)
            l2n = compare(image, resnetExample, 32)
            print(f'fail prediction for resnet: {failedpred}')
            print(f'Generations took to generate model: {resnetGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            modelGen += resnetGeneration
            modelL1 += l1n
            modelL2 += l2n
            totaltime += resnetEndTime


            collection.append(
                Result("KumarDNN", resnetGeneration, l1n, l2n, resnetImage, failedpred, resnetEndTime, image,
                       starting_index + i))

    print("\n\n\n")

    import pickle

    pickle_out = open("./vidnerovaGA/regularmodelresults.pickle", "wb")
    pickle.dump(collection, pickle_out)
    pickle_out.close()
    count = 1
    print(f"\nAverage Number of Generations for model - {modelGen/count}")
    print(f"Average Number of l1 for model - {modelL1/count}")
    print(f"Average Number of l2 for model - {modelL2/count}")
    print(f"Average Number of time for model - {totaltime/count}")

if __name__ == '__main__':
    import warnings

    import timeit
    start = timeit.default_timer()

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    parser = argparse.ArgumentParser()
    main(init(parser.parse_args()))

    stop = timeit.default_timer()
    print("\n\n\n\n")
    print("END OF PROGRAM EXECUTION")
    print('Total Time: ', stop - start)
