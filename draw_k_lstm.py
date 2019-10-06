import numpy as np
import pandas as pd
import time
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

filename ='results/1556659881.22mses.npy'

if __name__ == '__main__':

    total = np.load(filename)
    #total = np.asarray([[1.0, 2.0,3.0],[2.0,2.1,4],[3.0,1.9,6]])

    fig = plt.figure(figsize=(10, 6))
    x = np.linspace(5, 120, 24) #Change to x in k_lstm.py


    for i,mses in enumerate(total):

        plt.plot(x,mses, lw=1, alpha=0.5,
                 label='MSE runtime %d' % (i+1))

    mean_mse = np.mean(total, axis=0)
    std_mse = np.std(total,axis=0)

    plt.plot(x, mean_mse, color='b',
             label=r'Mean MSE',
             lw=2, alpha=.8)

    tprs_upper = mean_mse + std_mse
    tprs_lower = mean_mse - std_mse

    plt.fill_between(x, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Squared Error')
    plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
    plt.tight_layout()
    fig.savefig('draw_mses.png', dpi=300)