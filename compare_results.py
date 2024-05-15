import matplotlib.pyplot as plt
import numpy as np
import os
import torch

if __name__ == '__main__':
    """
    plot validation loss of all networks in the training folder
    """
    models = os.listdir("trainings")
    results = []
    fig,axs = plt.subplots(1)
    #map simple_cnn name to a model feature
    models = {i:i for i in models
              }
    start = 0
    stop = 100
    # plot losses
    for model,name in models.items():
        model_path = 'trainings/{}'.format(model)
        checkpoint = torch.load(model_path)
        results = checkpoint["loss"]#if not "mini" in model else sum(c[1])
        #print(name,np.around(checkpoint["loss"][stop][1]/50,1))
        axs.plot([sum(c[1])  for c in checkpoint["loss"][start:stop]],label=name)
    axs.set_title("tot loss")
    plt.tight_layout()
    plt.legend()
    #save figure as svg in figures folder
    #plt.savefig("figures/net_comparison.svg")
    plt.show()