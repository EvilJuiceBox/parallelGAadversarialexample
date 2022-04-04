from genetic_algorithm import *
from operator import itemgetter, attrgetter

class HighConfidenceTraditionalGeneticAlgorithm(GeneticAlgorithm):
    '''
    Generates a new adversarial example given the image and ground truth
    '''
    def generate(self, populationSize, generation, inputImage, model, y_truth, IMAGEDIMENSION):
        print("Generating an adversarial image using the tradition GA approach")
        population = []
        numberOfChildren = 3  # 3 children per generation.
        tournamentSize = 3  # tournament size

        # init population
        for i in range(populationSize):
            population.append(Candidate(inputImage))

        for i in range(generation):
            tempPopulation = copy.deepcopy(population)

            for j in range(numberOfChildren):
                offspring1, offspring2 = self.tournamentSelection(population, tournamentSize, model, y_truth)

                # Crossover operation
                if random.random() < 0.6:
                    child1, child2 = self.crossover(offspring1, offspring2, IMAGEDIMENSION)
                else:
                    child1, child2 = offspring1, offspring2

                # vidnerova original: 0.1, current: 0.5
                if random.random() < 0.1:
                    # Mutate operation and add to temp pop
                    self.mutate(child1, IMAGEDIMENSION)
                    self.mutate(child2, IMAGEDIMENSION)

                tempPopulation.append(Candidate(child1.getImage()))
                tempPopulation.append(Candidate(child2.getImage()))

            # cull population down to original size, and proceed to next gen.
            tempPopulation = self.calculatePopulationFitness(tempPopulation, model, y_truth)
            tempPopulation.sort(key=attrgetter("_fitness"), reverse=True)

            if tempPopulation[0].getFitness() == float("inf"):
                print("The solution was found at generation: " + str(i))
                return tempPopulation[0].getImage(), i

            population = self.survivorSelection(tempPopulation, populationSize, 3, model,
                                                y_truth)  # elitism of 3 per round, chosen arbitrary
            if i % 100 == 0:
                print("End of generation: " + str(i) + "; Best performing member: " + str(
                    population[0].getFitness()) + "; Worse performing member: " + str(
                    population[len(population) - 1].getFitness()))
            # print("END OF GENERATION: " + str(i))
        return population[0].getImage(), -1

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

        y_remove_truth = copy.deepcopy(y)
        prediction = y_remove_truth[0].tolist()
        del prediction[truth]  # remove the truth, then get argmax
        prediction = np.array(prediction)

        if truth != np.argmax(y) and prediction[np.argmax(prediction)] > 0.95:
            # print(f"np.argmax(pred): {np.argmax(prediction)}")
            # print(f"type: {type(np.argmax(prediction))}")
            # if truth != np.argmax(y) and y[0][np.argmax(y)] > 0.9:  # and argmax is greater than 90% confidence
            return float("inf") # -1

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
