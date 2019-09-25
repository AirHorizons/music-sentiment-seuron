import numpy as np

class GeneticAlgorithm:
    def __init__(self, neuron, neuron_ix, seq_data, popSize=100, mutRate=0.1, elitism=1, ofInterest=0):
        self.ofInterest   = ofInterest
        self.popSize      = popSize
        self.indSize      = len(neuron_ix)
        self.mutRate      = mutRate
        self.elitism      = elitism
        self.neuron       = neuron
        self.seq_data     = seq_data
        self.domain       = (-2, 2)
        self.neuron_ix    = neuron_ix
        self.inds = np.random.uniform(self.domain[0], self.domain[1], (popSize, self.indSize))

    def calcFitness(self, ind, experiments=30):
        label_guess = np.zeros(experiments)

        # Override neuron weights with the gens of the individual
        override_neurons = {}
        for i in range(self.indSize):
            n_ix = self.neuron_ix[i]
            override_neurons[n_ix] = ind[i]

        ini_seq = self.seq_data.str2symbols("\n")
        for i in range(experiments):
            print("Calc Fitness:", i)

            gen_seq, _ = self.neuron.generate_sequence(self.seq_data, ini_seq, 128, 1.0, override=override_neurons)
            guess = self.neuron.predict_sentiment(self.seq_data, [gen_seq])

            label_guess[i] = np.abs(guess - self.ofInterest)

        # Penalize this individual with the prediction error
        # validation_shard = "../input/generative/midi/vgmidi_shards/validation/vgmidi_11_shortest.txt"
        # error = self.neuron.evaluate(self.seq_data, 128, 256, validation_shard)

        fitness = sum(label_guess)/len(label_guess)
        return 1.0 - fitness

    def evaluate(self):
        fitness = np.zeros(self.popSize)
        for i in range(self.popSize):
            fitness[i] = self.calcFitness(self.inds[i])
        return fitness

    def cross(self, parents):
        nextPop = np.zeros_like(self.inds)

        for i in range(self.elitism):
            nextPop[i] = parents[i]

        for i in range(self.elitism, self.popSize):
            # one-point crossover
            pos = np.random.randint(0, self.indSize)

            # Take parents two-by-two
            p1 = parents[i-1]
            p2 = parents[i]

            # Create two children with crossover
            nextPop[i] = np.concatenate((p2[:pos], p1[pos:]))

        return nextPop

    def mutate(self, nextPop):
        for i in range(self.elitism, self.popSize):
            for j in range(self.indSize):
                if np.random.random() < self.mutRate:
                    nextPop[i][j] = np.random.uniform(self.domain[0], self.domain[1])

    def select(self, fitness):
        descending_args = np.argsort(-fitness)

        sorted_inds = self.inds[descending_args]
        sorted_fits = fitness[descending_args]

        matingPool = np.zeros_like(self.inds)

        for i in range(self.elitism):
            matingPool[i] = sorted_inds[i]

        for i in range(self.elitism, self.popSize):
            matingPool[i] = self.roullete_wheel(sorted_inds, sorted_fits)

        return matingPool

    def roullete_wheel(self, sorted_inds, sorted_fits):
        pick = np.random.uniform(0, sum(sorted_fits))

        fitnessSoFar = 0
        for i in range(self.popSize):
            fitnessSoFar += sorted_fits[i]

            if pick < fitnessSoFar:
                return sorted_inds[i]

        return sorted_inds[i]

    def get_best_individual(self, fitness):
        descending_args = np.argsort(-fitness)

        best_fit = fitness[descending_args[0]]
        best_ind = self.inds[descending_args[0]]

        print("best ind", best_ind)
        print("best fit", best_fit)
        return best_ind, best_fit

    def evolve(self, epochs=10):
        for i in range(epochs):
            print("-> Epoch", i)
            fitness = self.evaluate()
            self.get_best_individual(fitness)

            parents = self.select(fitness)
            nextPop = self.cross(parents)
            self.mutate(nextPop)

            self.inds = nextPop
            print("inds", self.inds)

        # return best individual
        return self.get_best_individual(fitness)
