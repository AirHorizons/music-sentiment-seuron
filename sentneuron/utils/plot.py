import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_logits(save_root, xs, ys, top_neurons):
    save_root = os.path.join(save_root, 'results')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Plotting logits at', save_root)

    for i, n in enumerate(top_neurons):
        plot_logit_and_save(xs, ys, n, os.path.join(save_root, str(i) + '_' + str(n)))

def plot_logit_and_save(xs, ys, neuron_index, name):
    sentiment_unit = xs[:,neuron_index]
    # plt.title('Distribution of Logit Values')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Value of the Sentiment Neuron')
    plt.hist(sentiment_unit[ys == 0], bins=25, alpha=0.5, label='Negative Reviews')
    plt.hist(sentiment_unit[ys == 1], bins=25, alpha=0.5, label='Positive Reviews')
    plt.legend()
    plt.savefig(name + '.png')
    plt.clf()

def plot_weight_contribs_and_save(save_root, coef):
    save_root = os.path.join(save_root,'results')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Plotting weights at', save_root)

    plt.title('Values of Resulting L1 Penalized Weights')
    plt.tick_params(axis='both', which='major')
    coef = normalize(coef)
    plt.plot(range(len(coef[0])), coef.T)
    plt.xlabel('Neuron (Feature) Index')
    plt.ylabel('Neuron (Feature) weight')
    plt.savefig(os.path.join(save_root, "weights"))
    plt.clf()

def plot_heatmap(save_root, preprocessed_text, values):
    save_root = os.path.join(save_root,'results')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    n_limit = 64
    num_chars = len(preprocessed_text)

    for i in np.arange(0, len(values), n_limit):
        if i + n_limit > num_chars:
            end_index = num_chars
        else:
            end_index = i+n_limit

        values_limited = values[i:end_index]
        values_reshaped = values_limited.reshape((1, end_index - i))

        chars_limited = list(preprocessed_text[i:end_index])
        chars_reshaped = np.array(chars_limited).reshape((1, end_index - i))

        fig, ax = plt.subplots(figsize=(20,0.5))
        ax = sns.heatmap(values_reshaped, annot=chars_reshaped, fmt='', annot_kws={"size":15}, vmin=-1, vmax=1, cmap='RdYlGn')

        plt.savefig(os.path.join(save_root, "heatmap_" + str(i)))
        plt.clf()

def normalize(coef):
    norm = np.linalg.norm(coef)
    coef = coef/norm
    return coef
