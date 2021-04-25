class Candidate():
    def __init__(self, my_image):
        self._image = my_image
        self.fitness = 0

    def getImage(self):
        return self._image

    def setImage(self, image):
        self._image = image

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness

    def __str__(self):
        result = str(type(self._image)) + "\n"
        result += "Fitness:" + str(self.fitness)
        return result

    def __repr__(self):
        return self.__str__()