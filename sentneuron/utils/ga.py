import numpy as np

class GeneticAlgorithm:
    def __init__(self, neuron, neuron_ix, seq_data, logreg, popSize=100, crossRate=0.95, mutRate=0.1, elitism=3, ofInterest=1.0):
        self.ofInterest   = ofInterest
        self.popSize      = popSize
        self.indSize      = len(neuron_ix)
        self.crossRate    = crossRate
        self.mutRate      = mutRate
        self.elitism      = elitism
        self.neuron       = neuron
        self.seq_data     = seq_data
        self.logreg       = logreg
        self.domain       = (-5, 5)
        self.neuron_ix    = neuron_ix
        self.inds = np.random.uniform(self.domain[0], self.domain[1], (popSize, self.indSize))
        print(self.inds)

    def isSilence(self, sequence):
        non_silence_symbs = 0
        for symb in sequence:
            if symb[0] == "n":
                non_silence_symbs += 1

        return non_silence_symbs == 0

    def calcFitness(self, ind, experiments=30):
        label_guess = []

        # Override neuron weights with the gens of the individual
        override_neurons = {}
        for i in range(len(self.neuron_ix)):
            n_ix = self.neuron_ix[i]
            override_neurons[n_ix] = ind[i]

        for i in range(experiments):
            ini_seq = self.seq_data.str2symbols(".")
            gen_seq = self.neuron.generate_sequence(self.seq_data, ini_seq, 256, 1.0, override=override_neurons)

            split = gen_seq.split(" ")
            split = list(filter(('').__ne__, split))

            # if self.isSilence(split):
            #     fitness.append(1.)
            # else:
            trans_seq, _ = self.neuron.transform_sequence(self.seq_data, split)
            guess = self.logreg.predict([trans_seq])[0]

            label_guess.append((guess - self.ofInterest)**2)

        # Penalize this individual with the prediction error
        validation_shard = "../input/generative/midi/vgmidi_shards/validation/vgmidi_11_shortest.txt"
        error = self.neuron.evaluate(self.seq_data, 128, 256, validation_shard)

        fitness = error + (sum(label_guess)/len(label_guess))
        return 2.0 - fitness
        # return (ind - self.ofInterest)**2

    def evaluate(self):
        fitness = []
        for i in range(self.popSize):
            print("---->", "Evaluating ind", i)
            fitness.append(self.calcFitness(self.inds[i]))
        return np.array(fitness)

    def cross(self, nextPop):
        for i in range(self.elitism, self.popSize - 1):
            if np.random.random() < self.crossRate:
                nextPop[i] = np.array((nextPop[i] + nextPop[i+1])/2.)

    def mutate(self, nextPop):
        for i in range(self.elitism, self.popSize):
            for gene in range(self.indSize):
                if np.random.random() < self.mutRate:
                    self.inds[i][gene] = np.random.uniform(self.domain[0], self.domain[1])

    def select(self, fitness):
        descending_args = np.argsort(-fitness)
        sorted_inds = list(self.inds[descending_args])
        sorted_fits = list(fitness[descending_args])

        nextPop = []
        for i in range(self.popSize):
            if i < self.elitism:
                nextPop.append(sorted_inds[i])
            else:
                nextPop.append(self.roullete_wheel(sorted_inds, sorted_fits))

        return np.array(nextPop)

    def roullete_wheel(self, sorted_inds, sorted_fits):
        pick = np.random.uniform(0, sum(sorted_fits))

        current = 0
        for i in range(self.popSize):
            current += sorted_fits[i]
            if current >= pick:
                return sorted_inds[i]

    def evolve(self, epochs=10):
        for i in range(epochs):
            print("-> Epoch", i)
            fitness = self.evaluate()
            nextPop = self.select(fitness)
            print(self.inds)
            print(fitness)
            self.cross(nextPop)
            self.mutate(nextPop)

            self.inds = nextPop

        # Get best individual
        fitness = self.evaluate()
        descending_args = np.argsort(-fitness)

        best_fit = fitness[descending_args][0]
        best_ind = self.inds[descending_args][0]

        print("best ind", best_ind)
        print("best fit", best_fit)

        return best_ind, best_fit
