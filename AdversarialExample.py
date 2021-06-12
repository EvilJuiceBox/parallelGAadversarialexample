class AdversarialExample():
    def __init__(self, advex):
        self._adversarialExample = advex
        self._mse = 0
        self.l1norm = 0
        self.generation = 0
        self.time = 0

    def setTime(self, t):
        self.time = t

    def getTime(self):
        return self.time

    def setGeneration(self, g):
        self.generation = g

    def getGeneration(self):
        return self.generation

    def setl1(self, l1fit):
        self.l1norm = l1fit

    def getl1(self):
        return self.l1norm

    def getImage(self):
        return self._adversarialExample

    def setImage(self, image):
        self._adversarialExample = image

    def setMSE(self, mse):
        self._mse = mse

    def getMSE(self):
        return self._mse

    def __str__(self):
        result = str(type(self._adversarialExample)) + "\n"
        result += "MSE:" + str(self._mse)
        return result

    def __repr__(self):
        return self.__str__()