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


# takes in an image and modifies the pixels located at x y to new rgb value
def modifyImageByPixel(image, x, y, newvalue):
    image[x][y] = newvalue


def mutate(candidate, IMAGEDIMENSION):
    target = candidate.getImage()
    for i in range(3):  # modify 3 random pixel, number is selected arbitrary for now
        randomPixelMutate(target, random.randint(0, IMAGEDIMENSION-1), random.randint(0, IMAGEDIMENSION-1))
    candidate.setImage(target)


def randomPixelMutate(image, x, y):
    channel = random.randint(0, 2)  # find a random channel to modify
    value = image[x][y][channel] * 255  # rgb channel value from 0-255

    # print("Original:" + str(value))

    if random.randint(0, 1) == 0:
        value = value + (255 * 0.1)  # increase pixel value by 10 percent
        if value > 255:
            value = 255
    else:
        value = value - (255*0.1)  # decrease value by 10 percent
        if value < 0:
            value = 0

    image[x][y][channel] = value/255

    # print("New:" + str(image[x][y][channel] * 255))
    # print(" ")


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
                mutate(target, IMAGEDIMENSION)
                tempPopulation.append(Candidate(target.getImage()))

            for j in range(numberOfChildren):
                offspring1, offspring2 = tournamentSelection(islands[z], tournamentSize, model, y_truth)
                # Crossover operation
                child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)

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
                return tempPopulation[0].getImage()

            islands[z] = survivorSelection(tempPopulation, populationSize, 3, model, y_truth)  # elitism of 3 per round, chosen arbitrary

        if i % 10 == 0:  # every 10 generation, migrate
            migrate = 0
            while migrate < populationSize:
                temp = islands[0][migrate]
                islands[0][migrate] = islands[1][migrate]
                islands[1][migrate] = islands[2][migrate]
                islands[2][migrate] = temp
                migrate += 1

        if i % 100 == 0:
            print("End of generation: " + str(i) + "; Best performing member: " + str(islands[0][0].getFitness()) + "; Worse performing member: " + str(islands[len(islands)-1][0].getFitness()))
        # print("END OF GENERATION: " + str(i))

    return getBestMember(islands).getImage()


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
            mutate(target, IMAGEDIMENSION)
            tempPopulation.append(Candidate(target.getImage()))

        for j in range(numberOfChildren):
            offspring1, offspring2 = tournamentSelection(population, tournamentSize, model, y_truth)
            # Crossover operation
            child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)

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
            return tempPopulation[0].getImage()

        population = survivorSelection(tempPopulation, populationSize, 3, model, y_truth)  # elitism of 3 per round, chosen arbitrary
        if i % 100 == 0:
            print("End of generation: " + str(i) + "; Best performing member: " + str(population[0].getFitness()) + "; Worse performing member: " + str(population[len(population)-1].getFitness()))
        # print("END OF GENERATION: " + str(i))
    return population[0].getImage()


# input image, tournament size (usually 3), returns 2 individuals that performs the best within the tournament selection
def tournamentSelection(inp, tournamentSize, model, truth):  # returns 2 individuals via tournament selection (cannot be same)
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


def calculatePopulationFitness(population, model, truth):
    temp = []
    for sample in population:
        item = Candidate(sample.getImage())
        item.setFitness(getFitness(sample.getImage(), model, truth))
        temp.append(item)
    return temp


def init(args): # loads the pretrained model
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
                mse += (image1[i][j][k] - image2[i][j][k])**2

    return mse


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

    # load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # scale images from 0-255 to 0-1 and add channel dimension
    x_train = np.float32(x_train) / 255
    x_test = np.float32(x_test) / 255
    print(f'Input shape: {x_train.shape}')

    # convert label into a one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(f'Output shape: {y_train.shape}')

    # test show figure 800
    # sample = x_test[800]

    # img = Image.fromarray(np.uint8(sample.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()

    #

    # # test crossover
    # child1, child2 = crossover(Candidate(sample), Candidate(x_test[333]), 32)
    # img = Image.fromarray(np.uint8(child1.getImage().reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()
    # img = Image.fromarray(np.uint8(child2.getImage().reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()

    # test mutation
    # sample = x_test[800]
    # img = Image.fromarray(np.uint8(sample.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()
    # testSubject = Candidate(sample)
    # for i in range(200):
    #     mutate(testSubject, 32)
    # img = Image.fromarray(np.uint8(testSubject.getImage().reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()



    original = x_test[404]
    groundtruth = y_test[404]
    print(np.argmax(y_test[404]))

    x = np.expand_dims(original, 0)
    y = model.predict(x)
    print(f'Prediction: {y}')
    print(f'Prediction: {class_labels[np.argmax(y)]}')
    print(f'Prediction: {np.argmax(y)}')
    print(f'Prediction: {y[0][np.argmax(y)]}')





    # # EXPERIMENT 1, generating 10 random adversarial images, then sort by MSE
    # img = Image.fromarray(np.uint8(original.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    # img.show()
    # img.save("./results/" + "original.png")
    #
    # collection = []
    # for i in range(5):
    #     result = generateImageGA(populationSize=25, generation=1000, inputImage=original, model=model,
    #                                      y_truth=np.argmax(groundtruth), IMAGEDIMENSION=32)
    #     temp = AdversarialExample(result)
    #     temp.setMSE(compare(original, result, 32))
    #     collection.append(temp)
    #
    # collection.sort(key=attrgetter("_mse"), reverse=False)
    #
    # for i in range(len(collection)):
    #     img = Image.fromarray(np.uint8(collection[i].getImage().reshape(input_shape) * 255), 'RGB').resize((128, 128))
    #     imgTitle = "Adversarial Input MSE " + str(round(collection[i].getMSE(), 2)*100)
    #
    #     print(str(imgTitle))
    #     img.show(title=imgTitle)
    #     img.save("./results/" + str(imgTitle) + ".png")
    #
    #     x = np.expand_dims(collection[i].getImage(), 0)
    #     y = model.predict(x)
    #     print(f'Prediction: {y}')
    #     print(f'Prediction: {class_labels[np.argmax(y)]}')
    #     print(f'Prediction: {np.argmax(y)}')  # ground turth
    #     print(f'Prediction: {y[0][np.argmax(y)]}')



# # EXPERIMENT 2, generating 10 random adversarial images with parallel GA, then sort by MSE
    img = Image.fromarray(np.uint8(original.reshape(input_shape) * 255), 'RGB').resize((128, 128))
    img.show()
    img.save("./parallelResults/" + "original.png")

    collection = []
    for i in range(5):
        result = parallelGA(populationSize=15, generation=1000, inputImage=original, model=model,
                                         y_truth=np.argmax(groundtruth), IMAGEDIMENSION=32)
        temp = AdversarialExample(result)
        temp.setMSE(compare(original, result, 32))
        collection.append(temp)

    collection.sort(key=attrgetter("_mse"), reverse=False)

    for i in range(len(collection)):
        img = Image.fromarray(np.uint8(collection[i].getImage().reshape(input_shape) * 255), 'RGB').resize((128, 128))
        imgTitle = "Adversarial Input MSE " + str(round(collection[i].getMSE(), 2)*100)

        print(str(imgTitle))
        img.show(title=imgTitle)
        img.save("./parallelResults/" + str(imgTitle) + ".png")

        x = np.expand_dims(collection[i].getImage(), 0)
        y = model.predict(x)
        print(f'Prediction: {y}')
        print(f'Prediction: {class_labels[np.argmax(y)]}')
        print(f'Prediction: {np.argmax(y)}')  # ground turth
        print(f'Prediction: {y[0][np.argmax(y)]}')
        print(" ")




if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    parser = argparse.ArgumentParser()
    main(init(parser.parse_args()))
