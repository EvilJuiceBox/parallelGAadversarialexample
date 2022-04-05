import argparse
import os
import time

import numpy as np
from keras.models import load_model

import AdvGenerator
import coursegrained_ga
import high_conf_cgga
import traditional_ga
import high_conf_ga
import finegrained_ga
from DeepNeuralNetwork import KumarDNN, DeepNeuralNetworkUtil
from ensembleMain import Result, saveResults


def main(args):
    singleModel = load_model("./models/" + "resnet20.h5")
    singleModel.summary()
    model = KumarDNN(singleModel)

    image = DeepNeuralNetworkUtil.loadImg("./fgGA_test/generatedExample1036deer.png")

    failedpred = model.predict(np.expand_dims(image, 0))

    print(f"The prediction for the image is: {failedpred}")
    print(f"Current prediction label {DeepNeuralNetworkUtil.getClassLabel(failedpred)} with confidence of {failedpred[0][np.argmax(failedpred)]}")

    import pickle

    file = open("advImg", "rb")
    img = pickle.load(file)
    file.close()

    print("===============")
    failedpred = model.predict(np.expand_dims(img, 0))

    print(f"The prediction for the image is: {failedpred}")
    print(
        f"Current prediction label {DeepNeuralNetworkUtil.getClassLabel(failedpred)} with confidence of {failedpred[0][np.argmax(failedpred)]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())