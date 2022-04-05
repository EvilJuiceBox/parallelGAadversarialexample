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

"""
generates adversarial examples with the traditional GA approach
filepath: filename of the model to be used, assumed to be kumar model
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateGAResults(filepath, resultfolderpath, skip=0, count=30):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    i = skip
    collection = []
    ga = traditional_ga.TraditionalGeneticAlgorithm()
    while total < count:
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
            gaStartTime = time.time()
            gaExample, gaGeneration = ga.generate(populationSize=100, generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            gaEndtime = time.time() - gaStartTime

            print(f"\tgaModelTime: {gaEndtime}\n")

            # pgaStartTime = time.time()
            # pgaExample, pgaGeneration = AdvGenerator.parallelGA(populationSize=7, generation=1000,
            #                                                               inputImage=image, model=model,
            #                                                               y_truth=groundtruth, IMAGEDIMENSION=32)
            # pgaEndtime = time.time() - pgaStartTime
            #
            # print(f"\tEnsembleModelTime: {pgaEndtime}\n")

            if gaGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "gaResult_test" + str(starting_index + i)
            gaImage = np.expand_dims(gaExample, 0)
            failedpred = model.predict(gaImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(gaExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, gaExample, 32)
            l2n = AdvGenerator.compare(image, gaExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {gaGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            collection.append(
                Result("GA", gaGeneration, l1n, l2n, gaImage, failedpred, gaEndtime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "gaalgorithmresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


"""
generates adversarial examples with the traditional GA approach
filepath: filename of the model to be used, assumed to be kumar model
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateParallelGAResults(filepath, resultfolderpath, skip=0, count=30):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    i = skip
    collection = []
    cgGA = coursegrained_ga.CourseGrainedGeneticAlgorithm()

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        # DeepNeuralNetworkUtil.showImg(image)

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction:
            # generate adversarial example here.
            total += 1
            pgaStartTime = time.time()
            pgaExample, pgaGeneration = cgGA.generate(populationSize=20, generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            pgaEndtime = time.time() - pgaStartTime

            print(f"\tpgaModelTime: {pgaEndtime}\n")

            if pgaGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "cggaResult_test" + str(starting_index + i)
            pgaImage = np.expand_dims(pgaExample, 0)
            failedpred = model.predict(pgaImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(pgaExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, pgaExample, 32)
            l2n = AdvGenerator.compare(image, pgaExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {pgaGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')
            print(f'Time: {pgaEndtime}')

            collection.append(
                Result("pGA", pgaGeneration, l1n, l2n, pgaImage, failedpred, pgaEndtime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "pgaalgorithmresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


def generateFineGrainedGA(filepath, resultfolderpath, skip=0, count=30):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    i = skip
    collection = []
    fgGA = finegrained_ga.FineGrainedGeneticAlgorithm()

    while total < count:
        i += 1

        print("Generating an adversarial example for test set[" + str(starting_index + i) + "].")
        image = x_test[starting_index + i]

        # DeepNeuralNetworkUtil.showImg(image)

        x = np.expand_dims(image, 0)
        groundtruth = np.argmax(y_test[starting_index + i])
        basemodelprediction = np.argmax(model.predict(x))

        print(f'groundtruth: {groundtruth}')
        print(f'basemodelprediction: {basemodelprediction}')
        # correct predictions on both ensemble and regular model
        if groundtruth == basemodelprediction:
            # generate adversarial example here.
            total += 1
            pgaStartTime = time.time()
            pgaExample, pgaGeneration, queryCount = fgGA.generate(populationSize=4, grid_size=3, overlap_region="L5", generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            pgaEndtime = time.time() - pgaStartTime

            print(f"\tpgaModelTime: {pgaEndtime}\n")

            if pgaGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "cggaResult_test" + str(starting_index + i)
            pgaImage = np.expand_dims(pgaExample, 0)
            failedpred = model.predict(pgaImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(pgaExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, pgaExample, 32)
            l2n = AdvGenerator.compare(image, pgaExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'Generations took to generate model: {pgaGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')
            print(f'Time: {pgaEndtime}')

            print(f"The query count for the adversarial example is {queryCount}")

            collection.append(
                Result("fgGA", pgaGeneration, l1n, l2n, pgaImage, failedpred, pgaEndtime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "pgaalgorithmresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")

"""
generates adversarial examples with the high Confidence
filepath: filename of the model to be used, assumed to be kumar model
skip: number of items to skip. If skip = 10, then the first item generated will be 1010
count: number of images to be generated, defaults to be 30
"""
def generateHighConfResults(filepath, resultfolderpath, skip=0, count=30):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    i = skip
    collection = []

    hcga = high_conf_cgga.HighConfidenceGeneticAlgorithm()

    while total < count:
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
            highConfTime = time.time()
            highConfExample, highConfGeneration = hcga.generate(populationSize=32, generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            highConfEndTime = time.time() - highConfTime

            print(f"\thighConfModelTime: {highConfEndTime}\n")

            if highConfGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "highConfResult_test" + str(starting_index + i)
            highConfImage = np.expand_dims(highConfExample, 0)
            failedpred = model.predict(highConfImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(highConfExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, highConfExample, 32)
            l2n = AdvGenerator.compare(image, highConfExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'fail prediction confidence for ensemble: {np.argmax(failedpred[0])}')
            print(f'Generations took to generate model: {highConfGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            collection.append(
                Result("HighConfGA", highConfGeneration, l1n, l2n, highConfImage, failedpred, highConfEndTime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "highconfalgorithmresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")

def generateTradHighConfResults(filepath, resultfolderpath, skip=0, count=30):
    singleModel = load_model("./models/" + filepath)
    singleModel.summary()
    model = KumarDNN(singleModel)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress tensorflow error output
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    i = skip
    collection = []

    hcga = high_conf_ga.HighConfidenceTraditionalGeneticAlgorithm()

    while total < count:
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
            highConfTime = time.time()
            highConfExample, highConfGeneration = hcga.generate(populationSize=100, generation=10000,
                                                                          inputImage=image, model=model,
                                                                          y_truth=groundtruth, IMAGEDIMENSION=32)
            highConfEndTime = time.time() - highConfTime

            print(f"\thighConfModelTime: {highConfEndTime}\n")

            if highConfGeneration == -1:  # skips the last loop if over 1000 generations (fail to create one)
                total -= 1
                continue

            imgTitle = "highConfResult_test" + str(starting_index + i)
            highConfImage = np.expand_dims(highConfExample, 0)
            failedpred = model.predict(highConfImage)
            imgTitle = "./" + resultfolderpath + "/" + str(imgTitle) + str(
                DeepNeuralNetworkUtil.getClassLabel(failedpred)) + ".png"
            DeepNeuralNetworkUtil.saveImg(highConfExample, image, imgTitle)

            print(imgTitle)

            l1n = AdvGenerator.getl1normdiff(image, highConfExample, 32)
            l2n = AdvGenerator.compare(image, highConfExample, 32)
            print(f'fail prediction for ensemble: {failedpred}')
            print(f'fail prediction confidence for ensemble: {np.argmax(failedpred[0])}')
            print(f'Generations took to generate model: {highConfGeneration}')
            print(f'L1 norm difference: {l1n}')
            print(f'L2 norm difference: {l2n}')

            collection.append(
                Result("HighConfGA", highConfGeneration, l1n, l2n, highConfImage, failedpred, highConfEndTime, image,
                       starting_index + i))

            print(f"Saving results up to adversarial example {starting_index + i}")
            saveResults(collection, resultfolderpath, "highconfalgorithmresult")  # dumps the result, overwrites each old file.
            print("\n---------------------------------------------")


def main(args):
    # generateGAResults("resnet20.h5", "100individualga", count=50)  # 5 nonmutually exclusive items
    # generateFineGrainedGA("resnet20.h5", "50individualfgga_3grid", count=50)
    generateParallelGAResults("resnet20.h5", "50individualpga_5island", count=50)  # 5 nonmutually exclusive items
    # generateHighConfResults("resnet20.h5", "50individualhga", count=50)  # 5 nonmutually exclusive items
    # generateTradHighConfResults("resnet20.h5", "50individualtradhga", count=50)

    # from keras.datasets import cifar10
    # from keras.utils import to_categorical
    # from keras.applications.resnet50 import preprocess_input, decode_predictions
    #
    # input_shape = (32, 32, 3)
    # num_classes = 10
    # class_labels = {
    #     0: 'airplane',
    #     1: 'automobile',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer',
    #     5: 'dog',
    #     6: 'frog',
    #     7: 'horse',
    #     8: 'ship',
    #     9: 'truck',
    # }
    #
    # # load CIFAR-10 dataset
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # for i in range(1030, 1045):
    #     pgaImage = np.expand_dims(x_test[i], 0)
    #     imgTitle = "./" + "original_images" + "/" + str(i) + ".png"
    #     img = Image.fromarray(np.uint8(pgaImage.reshape(input_shape)), 'RGB').resize(
    #         (32, 32))
    #     img.save(imgTitle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
