import os

def plot_logits(save_root, X, Y_pred, top_neurons):
    """plot logits and save to appropriate experiment directory"""
    save_root = os.path.join(save_root,'logit_vis')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('plotting_logits at', save_root)

    for i, n in enumerate(top_neurons):
        plot_logit_and_save(trXt, trY, n, os.path.join(save_root, str(i)+'_'+str(n)))


def plot_logit_and_save(logits, labels, logit_index, name):
    """
    Plots histogram (wrt to what label it is) of logit corresponding to logit_index.
    Saves plotted histogram to name.
    Args:
        logits:
        labels:
        logit_index:
        name:
"""
    logit = logits[:,logit_index]
    plt.title('Distribution of Logit Values')
    plt.ylabel('# of logits per bin')
    plt.xlabel('Logit Value')
    plt.hist(logit[labels < .5], bins=25, alpha=0.5, label='neg')
    plt.hist(logit[labels >= .5], bins=25, alpha=0.5, label='pos')
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()
