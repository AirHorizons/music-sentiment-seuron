import numpy as np
import sentneuron as sn

from .ga import GeneticAlgorithm

def evolve_weights(neuron, seq_data, results_path):
    n_not_zero = len(np.argwhere(neuron.sent_classfier.coef_))
    sentneuron_ixs = neuron.get_top_k_neuron_weights(k=n_not_zero)

    print("total sentneuron_ixs:", len(sentneuron_ixs))
    print(sentneuron_ixs)

    sn.utils.plot_logits(results_path, trXt, np.array(trY), sentneuron_ixs, fold="fold_")
    sn.utils.plot_weight_contribs_and_save(results_path, neuron.sent_classfier.coef_, fold="fold_")

    genAlg = GeneticAlgorithm(neuron, sentneuron_ixs, seq_data, ofInterest=0)
    best_ind, best_fit = genAlg.evolve()

    override = {}
    for i in range(len(sentneuron_ixs)):
        override[int(sentneuron_ixs[i])] = best_ind[i]

    print(override)
    with open('../output/ga_best.json', 'w') as fp:
        json.dump(override, fp)
