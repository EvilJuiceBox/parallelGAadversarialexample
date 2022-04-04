import numpy as np
# from keras.datasets import cifar10
from PIL import Image
import math
from copy import deepcopy


class Ensemble:
    mean = 120.70756512369792
    std = 64.1500758911213

    def __init__(self):
        self.submodels = []

    def append(self, submodel):
        self.submodels.append(submodel)

    def predict(self, x):
        # x = (x - self.mean) / (self.std + 1e-7)

        result = np.zeros(10)
        for i in range(len(self.submodels)):
            temp = self.submodels[i].predict(x)
            result = [a + b for a, b in zip(result, temp)]
        return np.true_divide(result, len(self.submodels))

    def evaluate(self, x_test, y_test):
        num_classes = 10

        correctCount = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, 0)
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correctCount += 1
        total = len(x_test)
        print("Out of " + str(total) + " test samples, the ensemble method correctly predicts " + str(correctCount) + " (" + str(correctCount/total) + ")")

    def recursive_evaluate(self, x_test, y_test):
        print(f"\nEvaluating each submodel's accuracy")
        for i in range(len(self.submodels)):
            correctCount = 0
            for x, y in zip(x_test, y_test):
                x = np.expand_dims(x, 0)
                prediction = self.submodels[i].predict(x)
                if np.argmax(prediction) == np.argmax(y):
                    correctCount += 1
            total = len(x_test)
            print("Out of " + str(total) + " test samples, the submodel {" + str(i) + "} correctly predicts " + str(
                correctCount) + " (" + str(correctCount / total) + ")")


class DeepNeuralNetworkUtil:
    @staticmethod
    def evenImage(image, beta=0.05):
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        reconstr = DeepNeuralNetworkUtil.evenout(r, beta)
        reconstg = DeepNeuralNetworkUtil.evenout(g, beta)
        reconstb = DeepNeuralNetworkUtil.evenout(b, beta)
        result = np.dstack((reconstr, reconstg, reconstb))
        return result

    @staticmethod
    def evenout(image, beta=0.05):
        # deepcopy
        result = deepcopy(image)
        original = deepcopy(image)

        # pixel to North, adds beta percent of
        result = result + beta * np.vstack((np.zeros(original.shape[1]), original[:-1, :]))

        # pixel to East
        result = result + beta * np.hstack((original[:, 1:], np.zeros((original.shape[0], 1))))

        # pixel to South
        result = result + beta * np.vstack((original[1:, :], np.zeros(original.shape[1])))

        # pixel to West
        result = result + beta * np.hstack((np.zeros((original.shape[0], 1)), original[:, :-1]))

        # pixel to NE
        result = result + beta * np.vstack((np.zeros((original.shape[1])), (np.hstack((original[:-1, 1:], np.zeros((original.shape[0]-1, 1))))) ))

        # pixel to SE
        result = result + beta * np.vstack((np.hstack((original[1:, 1:], np.zeros((original.shape[0]-1, 1)))), np.zeros((original.shape[1])) ))

        # pixel to SW
        result = result + beta * np.vstack((np.hstack((np.zeros((original.shape[0]-1, 1)), original[1:, :-1])) , np.zeros((original.shape[1])) ))

        # pixel to NW
        result = result + beta * np.vstack((np.zeros((original.shape[1])), np.hstack((np.zeros((original.shape[0]-1, 1)), original[:-1, :-1])) ))

        return result / (1 + 8*beta)

    @staticmethod
    def getImageFromDataset(index):
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
        return x_test[index]

    @staticmethod
    def show(newImage, IMAGEDIMENSION=32, standardised=True, title=""):
        input_shape = (IMAGEDIMENSION, IMAGEDIMENSION, 3)
        num_classes = 10
        if standardised:
            img = DeepNeuralNetworkUtil.inverseStandardisation(newImage)
        else:
            img = newImage

        img = Image.fromarray(np.uint8(img.reshape(input_shape)), 'RGB').resize(
            (IMAGEDIMENSION, IMAGEDIMENSION))
        img.show(title=title)


    @staticmethod
    def showImg(newImage, originalImage, IMAGEDIMENSION=32, standardised=True):
        input_shape = (32, 32, 3)
        num_classes = 10
        if standardised:
            img = DeepNeuralNetworkUtil.inverseStandardisation(newImage)
            originalImage = DeepNeuralNetworkUtil.inverseStandardisation(originalImage)
        else:
            img = newImage
            originalImage = originalImage

        # the loop ensures that each pixel value in the adversarial img will round away from the original image.
        for i in range(IMAGEDIMENSION):
            for j in range(IMAGEDIMENSION):
                for k in range(3):  # RGB channels
                    # print(img[i][j][k])
                    # print(originalImage[i][j][k])
                    if img[i][j][k] > originalImage[i][j][k]:
                        img[i][j][k] = math.ceil(img[i][j][k])
                    if img[i][j][k] < originalImage[i][j][k]:
                        img[i][j][k] = math.floor(img[i][j][k])

        img = Image.fromarray(np.uint8(img.reshape(input_shape)), 'RGB').resize(
            (32, 32))
        img.show()

        return None

    @staticmethod
    def predict(model, input):
        input = np.expand_dims(input, axis=0)
        y = model.predict(input)
        print(f'Prediction result for model: {y}')
        print(f'Predicted label: {DeepNeuralNetworkUtil.getClassLabel(y)}')
        print(f'Confidence for label: {y[0][np.argmax(y)]}')
        return y

    @staticmethod
    def inverseStandardisation(value):
        return value * (64.1500758911213 + 1e-7) + 120.70756512369792

    @staticmethod
    def standarisation(value):
        return (value - 120.70756512369792) / (64.1500758911213 + 1e-7)

    @staticmethod
    def loadImg(filepath, IMAGEDIMENSION=32):
        img = Image.open(filepath)
        # img.show()
        # img = img.convert('RGB')
        img = img.resize((IMAGEDIMENSION, IMAGEDIMENSION))
        x = np.asarray(img)
        # x = np.expand_dims(x, axis=0)  # x: [1, 32, 32, 1]
        return x

    @staticmethod
    def getClassLabel(prediction):
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
        return class_labels[np.argmax(prediction)]

    @staticmethod
    def getClassLabelByIndex(index):
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
        return class_labels[index]

    """
    Custom save function to save the image that rounds it away from the original image (more adversarial)
    """
    @staticmethod
    def saveImg(newImage, originalImage, title="untitled.png", IMAGEDIMENSION=32, standardised=True):
        input_shape = (32, 32, 3)
        num_classes = 10
        if standardised:
            img = DeepNeuralNetworkUtil.inverseStandardisation(newImage)
            originalImage = DeepNeuralNetworkUtil.inverseStandardisation(originalImage)
        else:
            img = newImage
            originalImage = originalImage

        # the loop ensures that each pixel value in the adversarial img will round away from the original image.
        for i in range(IMAGEDIMENSION):
            for j in range(IMAGEDIMENSION):
                for k in range(3):  # RGB channels
                    # print(img[i][j][k])
                    # print(originalImage[i][j][k])
                    if img[i][j][k] > originalImage[i][j][k]:
                        img[i][j][k] = math.ceil(img[i][j][k])
                    if img[i][j][k] < originalImage[i][j][k]:
                        img[i][j][k] = math.floor(img[i][j][k])

        img = Image.fromarray(np.uint8(img.reshape(input_shape)), 'RGB').resize(
            (32, 32))
        img.save(title)

        return None


class Resnet:
    def __init__(self, m):
        self.model = m

    def predict(self, x):
        # x = np.expand_dims(x, 0)
        x = np.float32(x) / 255
        return self.model.predict(x)


class KumarDNN:
    # mean and std for x train and test as non float32
    mean = 120.70756512369792
    std = 64.1500758911213

    # mean and std for x as float32
    # mean = 120.70748
    # std = 64.150024
    @staticmethod
    def inverseStandardisation(value):
        return value * (64.1500758911213 + 1e-7) + 120.70756512369792

    @staticmethod
    def standarisation(value):
        return (value - 120.70756512369792) / (64.1500758911213 + 1e-7)

    def __init__(self, m):
        self.model = m

    def predict(self, x):
        # x = np.expand_dims(x, 0)
        # x = (x - self.mean) / (self.std + 1e-7)
        return self.model.predict(x)

    def standardisePredict(self, x):
        x = (x - self.mean) / (self.std + 1e-7)
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        # x_test = (x_test - self.mean) / (self.std + 1e-7)
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))

    def fit(self, x_train, y_train, x_test, y_test):
        # train model
        from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

        # create learning rate plateau handler
        lr_plateau = ReduceLROnPlateau(
            monitor='loss',
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=5e-6
        )

        # create learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            lambda x:
            1e-3 * 5e-4 if x > 180
            else 1e-3 * 1e-3 if x > 160
            else 1e-3 * 1e-2 if x > 120
            else 1e-3 * 1e-1 if x > 80
            else 1e-3
        )

        print('Training...')
        # changing epoch from 100-200, batch size from 128 - 256
        # original paper" 256

        # 128, 200 epoch
        # loss: 1.438
        # accu: 0.799

        # 64, 125 epoch
        # Loss: 1.150
        # Accuracy: 0.824

        # standardised, 64, 125
        # Loss: 1.141
        # Accuracy: 0.827

        # 32, 200

        # Loss:     0.711
        # Accuracy: 0.884

        self.model.fit(x_train, y_train, batch_size=64, epochs=125, callbacks=[lr_plateau, lr_scheduler])

        # evaluate model
        print('Evaluating against test data...')
        result = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Loss:     {result[0]:0.3f}')
        print(f'Accuracy: {result[1]:0.3f}')

        # save model
        file_path = 'adversarial_training_dnn.h5'
        print(f'Saving {file_path}...')
        self.model.save(file_path)

    def self_evaluate(self, x_test, y_test):
        num_classes = 10

        correctCount = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, 0)
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correctCount += 1
        total = len(x_test)
        print("Out of " + str(total) + " test samples, the ensemble method correctly predicts " + str(correctCount) + " (" + str(correctCount/total) + ")")


class VGG:
    # mean and std for x train and test as non float32
    mean = 120.70756512369792
    std = 64.1500758911213

    # mean and std for x as float32
    # mean = 120.70748
    # std = 64.150024
    @staticmethod
    def inverseStandardisation(value):
        return value * (64.1500758911213 + 1e-7) + 120.70756512369792

    @staticmethod
    def standarisation(value):
        return (value - 120.70756512369792) / (64.1500758911213 + 1e-7)

    def __init__(self, m):
        self.model = m

    def predict(self, x):
        # x = np.expand_dims(x, 0)
        # x = (x - self.mean) / (self.std + 1e-7)
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        x_test = (x_test - self.mean) / (self.std + 1e-7)
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))


class DifferentEnsemble:
    mean = 120.70756512369792
    std = 64.1500758911213

    def __init__(self):
        import tensorflow as tf
        import os
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from keras.models import load_model

        self.submodels = []
        self.submodels.append(load_model("./models/hetero3/cifar10_resnet20.h5"))
        self.submodels.append(load_model("./models/hetero3/cifar10vgg.h5"))
        self.submodels.append(load_model("./models/hetero3/kumardnn.h5"))


    def append(self, submodel):
        self.submodels.append(submodel)

    @staticmethod
    def inverseStandardisation(value):
        return value * (64.1500758911213 + 1e-7) + 120.70756512369792

    @staticmethod
    def standardisation(value):
        return (value - 120.70756512369792) / (64.1500758911213 + 1e-7)

    def predict(self, x):
        prediction = (0.777 * self.submodels[0].predict(x) + 0.9335 * self.submodels[1].predict(x) + 0.8907 * self.submodels[2].predict(x)) / (0.777 + 0.9335 + 0.8907)
        return prediction

    def predict_standardised(self, x):
        x_std = self.standardisation(x)
        prediction = (0.777 * self.submodels[0].predict(x_std) + 0.9335 * self.submodels[1].predict(x_std) + 0.8907 * self.submodels[2].predict(x_std)) / (0.777 + 0.9335 + 0.8907)
        return prediction

    def evaluate(self, x_test, y_test):
        correctCount = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, 0)
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correctCount += 1
        total = len(x_test)
        print("Out of " + str(total) + " test samples, the ensemble method correctly predicts " + str(
            correctCount) + " (" + str(correctCount / total) + ")")
