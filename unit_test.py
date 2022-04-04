from unittest.mock import patch

import genetic_algorithm
import argparse

@patch.multiple(genetic_algorithm.GeneticAlgorithm, __abstractmethods__=set())
def test_mutate():
    from Candidate import Candidate
    arr = [[[0,1,2], [3,4,5], [6,7,8]], [[10,11,12], [13,14,15], [16,17,18]], [[20,21,22], [23,24,25], [26,27,28]]]
    obj = Candidate(arr)
    obj.setImage(arr)
    class MockObject(genetic_algorithm.GeneticAlgorithm):
        pass
    mo = MockObject()
    mo.mutate(obj, 3)
    print(obj.getImage())

@patch.multiple(genetic_algorithm.GeneticAlgorithm, __abstractmethods__=set())
def test_crossover():
    from Candidate import Candidate
    arr = [[0,1,2], [3,4,5], [6,7,8]]
    arr2 = [[10,11,12], [13,14,15], [16,17,18]]

    parent1 = Candidate(arr)
    parent1.setImage(arr)
    parent2 = Candidate(arr2)
    parent2.setImage(arr2)

    class MockObject(genetic_algorithm.GeneticAlgorithm):
        pass

    mo = MockObject()
    child1, child2 = mo.crossover(parent1, parent2, 3)
    print2dmatrix(child1.getImage())
    print(" ")
    print2dmatrix(child2.getImage())


def print2dmatrix(arr):
    for i in range(len(arr)):
        print('[', end = "")
        for j in range(len(arr[0])):
            print('{0:02d}'.format(int(arr[i][j])), end=", ")
        print(']')

def main(args):
    # test_mutate()
    test_crossover()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
