from genetic_algorithm import *
from operator import itemgetter, attrgetter

class FineGrainedGeneticAlgorithm(GeneticAlgorithm):
    '''
    Generates a new adversarial example given the image and ground truth, input image is assumed to be standardised
    '''
    def generate(self, populationSize, grid_size, overlap_region, generation, inputImage, model, y_truth, IMAGEDIMENSION):
        print("fine grain ga start.")
        numberOfChildren = 3  # 3 children per generation.
        tournamentSize = 3  # tournament size
        grid = []

        queryCount = 0

        # random init for each island
        for i in range(grid_size):  # create n x n grid of populations
            inner_grid = []
            for j in range(grid_size):
                population = []
                for k in range(populationSize):
                    population.append(Candidate(inputImage))
                inner_grid.append(population)
            grid.append(inner_grid)

        overlapped_population = []
        for i in range(generation):
            if overlap_region == "L5":
                overlapped_population = self.getL5Population(grid)
            if overlap_region == "C9":
                # overlapped_population = getC9Population(grid)
                pass

            for rows in range(grid_size):
                for z in range(grid_size):
                    # save the current pop to tempPopulation
                    tempPopulation = grid[rows][z]

                    for j in range(numberOfChildren):
                        # Choose two parents to produce offsprings
                        offspring1, offspring2 = self.tournamentSelection(overlapped_population[rows][z], tournamentSize, model, y_truth)
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
                        tempPopulation.append(Candidate(child1.getImage()))
                        tempPopulation.append(Candidate(child2.getImage()))

                    # cull population down to original size, and proceed to next gen.
                    tempPopulation = self.calculatePopulationFitness(tempPopulation, model, y_truth)
                    queryCount += len(tempPopulation)
                    tempPopulation.sort(key=attrgetter("_fitness"), reverse=True)

                    if tempPopulation[0].getFitness() == float("inf"):
                        print("The solution was found at generation: " + str(i))
                        return tempPopulation[0].getImage(), i, queryCount

                    grid[rows][z] = self.survivorSelection(tempPopulation, populationSize, 3, model,
                                                   y_truth)  # elitism of 3 per round, chosen arbitrary


            if i % 100 == 0:
                print("End of generation: " + str(i) + "; Best performing member: " + str(
                    grid[0][0][0].getFitness()) + "; Worse performing member: " + str(
                    grid[grid_size-1][grid_size - 1][0].getFitness()))
            # print("END OF GENERATION: " + str(i))

        return self.getBestMember(grid[0]).getImage(), -1, queryCount

    def getL5Population(self, grid):
        overlapping_population = copy.deepcopy(grid)
        N = len(grid)

        for r in range(N):
            for c in range(N):
                if r > 0:  ## up
                    overlapping_population += grid[r-1][c]
                if r < N - 1:  ## down
                    overlapping_population += grid[r+1][c]
                if c > 0:  ## left
                    overlapping_population += grid[r][c-1]
                if c < N - 1:  ## right
                    overlapping_population += grid[r][c+1]

        return overlapping_population

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
            return float("inf") # -1

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
