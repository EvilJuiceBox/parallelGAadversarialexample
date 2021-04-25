class AdversarialExample():
    def __init__(self, advex):
        self._adversarialExample = advex
        self._mse = 0

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