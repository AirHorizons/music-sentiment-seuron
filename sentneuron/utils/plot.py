import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

def plot_heatmap(save_root, text, neuron_values, polarity=1):
    save_root = os.path.join(save_root,'results')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Plotting weights at', save_root)

    n_limit = 74

    cell_height = .325
    cell_width  = .15

    text = list(map(lambda x: x.replace('\n', '\\n'), text))

    num_chars = len(text)
    total_chars = math.ceil(num_chars/float(n_limit)) * n_limit

    mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    text = np.array(text+[' ']*(total_chars-num_chars))

    neuron_values = np.array(neuron_values+[0]*(total_chars-num_chars))
    neuron_values *= polarity

    neuron_values = neuron_values.reshape(-1, n_limit)
    text = text.reshape(-1, n_limit)
    mask = mask.reshape(-1, n_limit)

    num_rows = len(neuron_values)
    plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
    sns.heatmap(neuron_values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn', xticklabels=False, yticklabels=False, cbar=False)
    plt.tight_layout()

    plt.savefig(os.path.join(save_root, "heatmap"))
    plt.clf()

def normalize(coef):
    norm = np.linalg.norm(coef)
    coef = coef/norm
    return coef
