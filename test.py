from Candidate import Candidate
from AdversarialExample import AdversarialExample
from operator import itemgetter, attrgetter

def main():
    # temp = 5
    # array = []
    # for i in range(5):
    #     array.append(temp)
    #
    # print(array)
    # array[4] = 10
    # print(array)
    #
    # candidate = Candidate("image")
    # print(candidate)

    # array = []
    # for i in range(5):
    #     array.append(i)
    #
    # print(len(array))
    # for i in range(len(array)):
    #     print(array[i])

    ae = AdversarialExample("img")
    ae.setMSE(12)
    ae1 = AdversarialExample("img")
    ae1.setMSE(100)
    ae2 = AdversarialExample("img")
    ae2.setMSE(20)

    array = []
    array.append(ae)
    array.append(ae1)
    array.append(ae2)

    array.sort(key=attrgetter("_mse"), reverse=False)

    print(array)


if __name__ == '__main__':
    main()