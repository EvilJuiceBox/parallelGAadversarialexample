from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import math
from PIL import Image

class Candidate():
    def __init__(self, my_image):
        self._image = my_image
        self._fitness = 0

    def getImage(self):
        return self._image

    def setImage(self, image):
        self._image = image

    def setFitness(self, fitness):
        self._fitness = fitness

    def getFitness(self):
        return self._fitness

    def __str__(self):
        result = str(type(self._image)) + "\n"
        result += "Fitness:" + str(self._fitness)
        return result

    def __repr__(self):
        return self.__str__()

class GeneticAlgorithm(ABC):
    """
    Function to choose survivours from the population. Elitism is maintained by providing how many elites must survive
    """
    @abstractmethod
    def survivorSelection(self, inputPopulation, cullsize, elitism, model, y):
        pass

    @abstractmethod
    def getFitness(self, image, model, truth):
        pass

    '''
    input image, tournament size (usually 3), returns 2 individuals that performs the best within the tournament selection
    '''
    @abstractmethod
    def tournamentSelection(self, inp, tournamentSize, model, truth):
        pass

    """
    Iteratively calculates the fitness for every individual in the population
    """
    @abstractmethod
    def calculatePopulationFitness(self, population, model, truth):
        pass


    def mutate(self, candidate, IMAGEDIMENSION):
        target = candidate.getImage()
        self.randomPixelMutate(target, random.randint(0, IMAGEDIMENSION - 1), random.randint(0, IMAGEDIMENSION - 1))
        candidate.setImage(target)

    # modified to use standardised cifar10 values
    def randomPixelMutate(self, image, x, y):
        channel = random.randint(0, 2)  # find a random channel to modify

        # commented out 255 due to stdisation of second dnn.
        value = image[x][y][channel]  # * 255  # rgb channel value from 0-255
        if random.randint(0, 1) == 0:
            value = value + (0.0993763439) #2.5 percent # increase pixel value by 10 percent (0.19875268739766835)

            if value > 2.093410378732659:
                value = 2.093410378732659
        else:
            value = value - (0.0993763439)  # decrease value by 2.5 percent  0.0993763439
            # 198752687 = 5%
            # 0.3975053754 = 10%
            if value < -1.8816433692207077:
                value = -1.8816433692207077

        image[x][y][channel] = value  # / 255

    '''
    Crossover function that swaps the pixel values of two images past random x and y point.
    '''
    def crossover(self, item1, item2, imageDimension):
        crossover_pointx = random.randint(0, imageDimension)
        crossover_pointy = random.randint(0, imageDimension)
        # print(str(crossover_pointx) + " :: " + str(crossover_pointy))
        temp1 = item1.getImage()
        temp2 = item2.getImage()

        for i in range(crossover_pointx, imageDimension):
            for j in range(crossover_pointy, imageDimension):
                tempPixel = copy.deepcopy(temp1[i][j])
                temp1[i][j] = temp2[i][j]
                temp2[i][j] = tempPixel

        item1.setImage(temp1)
        item2.setImage(temp2)

        return Candidate(copy.deepcopy(temp1)), Candidate(copy.deepcopy(temp2))

    """
    Removes an individual from the population
    """
    def removeIndividual(self, iterable, item):
        for i, o in enumerate(iterable):
            # if (o.getImage() == item.getImage()).all():
            if np.array_equal(o.getImage(), item.getImage()):
                del iterable[i]
                break
        return iterable

    # calculates the difference between the two images, average pixel value difference.
    def compare(self, image1, image2, IMAGEDIMENSION):
        mse = 0
        for i in range(IMAGEDIMENSION):
            for j in range(IMAGEDIMENSION):
                for k in range(3):  # RGB channels
                    mse += (int(image1[i][j][k]) - int(image2[i][j][k])) ** 2  # divided by 255 for new dnn

        return math.sqrt(mse)

    # returns the number of changed pixels
    def getl1normdiff(self, original, perturbed, IMAGEDIMENSION):
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

    def showImg(self, imgToDisplay):
        input_shape = (32, 32, 3)
        img = Image.fromarray(np.uint8(imgToDisplay.reshape(input_shape) * 255), 'RGB').resize((128, 128))
        img.show()

    def getDifference(self, original, perturbed, IMAGEDIMENSION):
        result = copy.deepcopy(original)
        for i in range(IMAGEDIMENSION):
            for j in range(IMAGEDIMENSION):
                for k in range(3):  # RGB channels
                    result[i][j][k] -= perturbed[i][j][k]
        return result
