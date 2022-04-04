from genetic_algorithm import *
from operator import itemgetter, attrgetter
import multiprocessing


class ParallelAlgorithm(GeneticAlgorithm):
    def gacycle(self, populationSize, generation, model, y_truth, queue, solution, tournamentSize, numberOfChildren,
                IMAGEDIMENSION):
        import tensorflow as tf
        import keras
        island_population = queue.get()
        for i in range(generation):
            print(f"Trying generation {i}")
            for j in range(numberOfChildren):
                # Choose two parents to produce offsprings
                offspring1, offspring2 = self.tournamentSelection(island_population, tournamentSize, model, y_truth)
                # Crossover operation
                # child1, child2 = self.crossover(offspring1, offspring2, IMAGEDIMENSION)

                if random.random() < 0.6:
                    child1, child2 = self.crossover(offspring1, offspring2, IMAGEDIMENSION)
                else:  # no crossover
                    child1, child2 = offspring1, offspring2
                # # Mutate operation and add to temp pop
                # mutate(child1, IMAGEDIMENSION)
                # mutate(child2, IMAGEDIMENSION)

                # vidnerova original: 0.1, current: 0.5
                if random.random() < 0.5:
                    # Mutate operation and add to temp pop
                    self.mutate(child1, IMAGEDIMENSION)
                    self.mutate(child2, IMAGEDIMENSION)
                # child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)
                #
                # mutate(child1, IMAGEDIMENSION)
                # mutate(child2, IMAGEDIMENSION)
                island_population.append(Candidate(child1.getImage()))
                island_population.append(Candidate(child2.getImage()))

            # cull population down to original size, and proceed to next gen.
            island_population = self.calculatePopulationFitness(island_population, model, y_truth)
            island_population.sort(key=attrgetter("_fitness"), reverse=True)

            if island_population[0].getFitness() == float("inf"):
                print("The solution was found at generation: " + str(i))
                solution.put(island_population[0].getImage())
                solution.put(i)
                return island_population[0].getImage(), i

            island_population = self.survivorSelection(island_population, populationSize, 3, model,
                                                       y_truth)  # elitism of 3 per round, chosen arbitrary
        queue.put(island_population)
        return None

    '''
    Generates a new adversarial example given the image and ground truth, input image is assumed to be standardised
    '''

    def generate(self, populationSize, generation, inputImage, model, y_truth, IMAGEDIMENSION):
        print("parallelGA start, best member migrates every 10 generations.")
        islandCount = 3  # 3 island, might allow dynamic allocation later
        numberOfChildren = 3  # 3 children per generation.
        tournamentSize = 3  # tournament size
        islands = []
        migrationCycles = 10
        solutions = []

        # islands = []
        # random init for each island
        for i in range(islandCount):
            population = []
            for j in range(populationSize):
                population.append(Candidate(inputImage))
            q = multiprocessing.Queue()
            q.put(population)
            islands.append(q)
            solutions.append(multiprocessing.Queue())
            # islands.append(population)
        print("Successfully initialised all islands and queues, beginning evolutionary search")
        for i in range(0, generation, migrationCycles):
            processes = []
            for z in range(islandCount):
                processes.append(multiprocessing.Process(target=self.gacycle,
                                                         args=(populationSize, migrationCycles, model, y_truth,
                                                               islands[z], solutions[z], tournamentSize,
                                                               numberOfChildren, IMAGEDIMENSION)))

                for p in processes:
                    p.start()

                for p in processes:
                    p.join()
                print("processes have joined")
                # tempPopulation = copy.deepcopy(islands[z])
                #
                # for j in range(numberOfChildren):
                #     # Choose two parents to produce offsprings
                #     offspring1, offspring2 = self.tournamentSelection(islands[z], tournamentSize, model, y_truth)
                #     # Crossover operation
                #     # child1, child2 = self.crossover(offspring1, offspring2, IMAGEDIMENSION)
                #
                #     if random.random() < 0.6:
                #         child1, child2 = self.crossover(offspring1, offspring2, IMAGEDIMENSION)
                #     else:  # no crossover
                #         child1, child2 = offspring1, offspring2
                #     # # Mutate operation and add to temp pop
                #     # mutate(child1, IMAGEDIMENSION)
                #     # mutate(child2, IMAGEDIMENSION)
                #
                #     # vidnerova original: 0.1, current: 0.5
                #     if random.random() < 0.5:
                #         # Mutate operation and add to temp pop
                #         self.mutate(child1, IMAGEDIMENSION)
                #         self.mutate(child2, IMAGEDIMENSION)
                #     # child1, child2 = crossover(offspring1, offspring2, IMAGEDIMENSION)
                #     #
                #     # mutate(child1, IMAGEDIMENSION)
                #     # mutate(child2, IMAGEDIMENSION)
                #     tempPopulation.append(Candidate(child1.getImage()))
                #     tempPopulation.append(Candidate(child2.getImage()))
                #
                # # cull population down to original size, and proceed to next gen.
                # tempPopulation = self.calculatePopulationFitness(tempPopulation, model, y_truth)
                # tempPopulation.sort(key=attrgetter("_fitness"), reverse=True)
                #
                # if tempPopulation[0].getFitness() == float("inf"):
                #     print("The solution was found at generation: " + str(i))
                #     return tempPopulation[0].getImage(), i
                #
                # islands[z] = self.survivorSelection(tempPopulation, populationSize, 3, model,
                #                                     y_truth)  # elitism of 3 per round, chosen arbitrary

                for sol in solutions:
                    if not sol.empty():
                        return sol.get(), sol.get()

            # if i % migrationCycles == 0:  # every 10 generation, migrate
            migrate = 0  # keep the best performing member of the island on its own island
            islandlist = []
            for iscnt in range(islandCount):
                tem = islands[iscnt].get()
                islandlist.append(tem)
                islands[iscnt].put(tem)

            # circular rotation
            temp = islandlist[0][migrate]
            for migrateIslandNum in range(islandCount-1):
                islandlist[migrateIslandNum][migrate] = islandlist[migrateIslandNum+1][migrate]
            islandlist[islandCount-1][migrate] = temp

            # print output every 100 genertions
            if i % 100 == 0:
                print("End of generation: " + str(i) + "; Best performing member: " + str(
                    islandlist[0][0].getFitness()) + "; Worse performing member: " + str(
                    islandlist[len(islandlist) - 1][0].getFitness()))
            # print("END OF GENERATION: " + str(i))

        islandlist = []
        for iscnt in range(islandCount):
            islandlist.append(islands[iscnt].get())
        return self.getBestMember(islandlist).getImage(), -1

    """
    Returns the best member among the three islands
    """
    def getBestMember(self, islands):
        best = islands[0][0]  # default first item of first islands
        for i in range(len(islands)):
            if best.getFitness() < islands[i][0].getFitness():
                best = islands[i][0]
        return best

    """
    Function to choose survivours from the population. Elitism is maintained by providing how many elites must survive
    """
    def survivorSelection(self, inputPopulation, cullsize, elitism, model, y):
        population = copy.deepcopy(inputPopulation)

        temp = []
        for i in range(cullsize):  # pick the remaining survivors
            # for i in range(cullsize - elitism):  # pick the remaining survivors
            if len(population) < 4:
                survivor = population[0]
                temp.append(survivor)
                population = self.removeIndividual(population, survivor)  # prevent same individual from appearing twice
            else:
                survivor, survivor1 = self.tournamentSelection(population, 3, model, y)
                temp.append(survivor)
                population = self.removeIndividual(population, survivor)  # prevent same individual from appearing twice

        temp = self.calculatePopulationFitness(temp, model, y)
        temp.sort(key=attrgetter("_fitness"), reverse=True)

        for j in range(elitism):
            if inputPopulation[j] not in temp:
                temp.insert(0, inputPopulation[j])
                temp.pop()

        # print(len(temp))
        return temp

    '''
    Returns the highest value that is not the ground truth
    This implies that individuals with higher fitness score is more fit.
    # truth: single value (e.g., 8)
    # y: [[confidence values...]]
    # np.argmax(y) returns current highest prediction
    # y[0][np.argmax(y)]: Confidence value of the highest prediction
    '''

    def getFitness(self, image, model, truth):
        # image = candidateInput.getImage()
        x = np.expand_dims(image, 0)
        y = model.predict(x)

        if truth != np.argmax(y):
            # if truth != np.argmax(y) and y[0][np.argmax(y)] > 0.9:  # and argmax is greater than 90% confidence
            return float("inf")  # -1

        prediction = y[0].tolist()
        del prediction[truth]  # remove the truth, then get argmax
        prediction = np.array(prediction)

        return prediction[np.argmax(prediction)]
        # returns the confidence level of the corresponding item
        # print("fitness estimate:")
        # print(y[0][np.argmax(y)])
        # return y[0][np.argmax(y)]

    '''
    input image, tournament size (usually 3), returns 2 individuals that performs the best within the tournament selection
    '''

    def tournamentSelection(self, inp, tournamentSize, model, truth):
        individuals = copy.deepcopy(inp)

        # individuals = self.calculatePopulationFitness(individuals, model, truth)
        round1 = random.sample(individuals, tournamentSize)

        result1 = round1[0]
        for i in range(len(round1)):
            if round1[i].getFitness() > result1.getFitness():
                result1 = round1[i]  # return the greatest individual from round1 sample

        individuals.remove(result1)  # remove round 1 individual

        round2 = random.sample(individuals, tournamentSize)

        result2 = round2[0]
        for i in range(len(round2)):
            if round2[i].getFitness() > result2.getFitness():
                result2 = round2[i]  # return the greatest individual from round1 sample

        return result1, result2

    """
    Iteratively calculates the fitness for every individual in the population
    """

    def calculatePopulationFitness(self, population, model, truth):
        temp = []
        for sample in population:
            item = Candidate(sample.getImage())
            item.setFitness(self.getFitness(sample.getImage(), model, truth))
            temp.append(item)
        return temp
