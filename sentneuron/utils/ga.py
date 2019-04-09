import numpy as np

class GeneticAlgorithm:
    def __init__(self, neuron, neuron_ix, seq_data, logreg, popSize=100, indSize=1, crossRate=0.9, mutRate=0.05, ofInterest=1.0):
        self.ofInterest   = ofInterest
        self.popSize      = popSize
        self.indSize      = indSize
        self.crossRate    = crossRate
        self.mutRate      = mutRate
        self.neuron       = neuron
        self.seq_data     = seq_data
        self.logreg       = logreg
        self.neuron_ix    = neuron_ix
        self.fits = np.random.uniform(-1, 1, (popSize, indSize))
        print(self.fits)

    def calcFitness(self, ind, experiments=30):
        fitness = []
        for i in range(experiments):
            ini_seq = self.seq_data.str2symbols("t_124")
            gen_seq = self.neuron.generate_sequence(self.seq_data, ini_seq, 128, 1.0, override={self.neuron_ix: ind})

            split = gen_seq.split(" ")
            split = list(filter(('').__ne__, split))
            trans_seq, _ = self.neuron.transform_sequence(self.seq_data, split)

            guess = self.logreg.predict([trans_seq])[0]
            fitness.append((guess - self.ofInterest)**2)

        return 1./(sum(fitness) + 1.)

    def evaluate(self):
        fitness = []
        for i in range(self.popSize):
            print("fits i", self.fits[i])
            fitness.append(self.calcFitness(self.fits[i][0]))
        return np.array(fitness)

    def cross(self, nextPop):
        for i in range(10, self.popSize):
            parents = nextPop[np.random.random_integers(0,9,2)]

            if (i * np.random.random()) > self.crossRate:
                nextPop[i] = np.array((parents[0] + parents[1])/len(parents))
            else:
                nextPop[i] = parents[0]

    def mutate(self, nextPop):
        for i in range(10, self.popSize):
            for gene in range(self.indSize):
                if np.random.random() < self.mutRate:
                    print("mutate")
                    self.fits[i][gene] = np.random.uniform(-1, 1)

    def select(self, fitness):
        nextPop = []
        for i in range(self.popSize):
            pick = np.random.uniform(0, sum(fitness))

            current = 0
            for i in range(self.popSize):
                current += fitness[i]
                if current > pick:
                    nextPop.append(self.fits[i])
                    break

        print("pop len", len(nextPop))

        return np.array(nextPop)

    def evolve(self, epochs=100):
        for i in range(epochs):
            print("-> Epoch", i)
            fitness = self.evaluate()
            print(fitness)

            nextPop = self.select(fitness)
            print(nextPop)

            self.cross(nextPop)
            self.mutate(nextPop)

            self.fits = nextPop
            print(self.fits)

        print("end", self.fits)
