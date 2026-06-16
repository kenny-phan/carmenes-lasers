import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'cm'          # Computer Modern serif
mpl.rcParams['mathtext.rm'] = 'serif'

plt.rcParams.update({'axes.linewidth' : 1.5, 
                     'ytick.major.width' : 1.5,
                     'ytick.minor.width' : 1.5,
                     'xtick.major.width' : 1.5,
                     'xtick.minor.width' : 1.5,
                     'xtick.labelsize': 12, 
                     'ytick.labelsize': 12,
                     'axes.labelsize': 18,
                     'axes.labelpad' : 5,
                     'axes.titlesize' : 24,
                     'axes.titlepad' : 0,
                     'font.family': 'Serif'
                    })
plt.style.use('tableau-colorblind10')
tableau_cb10 = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

def debug_print(verbose, *args):
    if verbose:
        print(*args)
        
def plot_spectra_elike(fig, axs, x, y, n_sections, title=None, xlabel=None, ylabel=None):
    x_sections = np.array_split(x, n_sections)
    y_sections = np.array_split(y, n_sections)

    axs[-1].set(xlabel=xlabel if xlabel is not None else None)
    axs[(n_sections//2)].set(ylabel=ylabel if ylabel is not None else None)
    fig.suptitle(title if title is not None else None)

    for i in range(n_sections):
        axs[i].plot(x_sections[i], y_sections[i])