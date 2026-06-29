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

def plot_spectra_obs(n_obs_range, wave_arr, spec_arr, sigma_arr, date_arr, ordidx, title=None, figsize=(20,24)):
    start, end = n_obs_range  # end is exclusive
    idxs = range(start, min(end, spec_arr.shape[2]))  # clamp to available obs
    n_obs = len(list(idxs))
    if n_obs <= 0:
        print("No observations to plot!")
        return
    if n_obs > 20:
        print("Too many observations for one figure!")
        return

    fig, axs = plt.subplots(n_obs, figsize=figsize)
    if n_obs == 1:
        axs = [axs]

    for ax_i, i in enumerate(idxs):
        wave = wave_arr[ordidx, :, i]
        spectra = spec_arr[ordidx, :, i]
        sigma = sigma_arr[ordidx, :, i]

        axs[ax_i].plot(wave, spectra, label=f"{date_arr[i]}")
        axs[ax_i].fill_between(wave, spectra - sigma, spectra + sigma,
                                alpha=0.5, color="orange", label="$1\\sigma$")
        axs[ax_i].legend(loc="upper left")

    if title is not None:
        fig.suptitle(title)

    return fig, axs

def compare_high_std_spectra(spec_arr, wave_arr, date_arr, high_std_mask, order_idx=None, obs_idx=None, verbose=False):
    # Find all flagged positions
    flagged_orders, flagged_obs = np.where(high_std_mask)

    if verbose: 
        print("\nFlagged observations:")
        print("=" * 40)
        for i, (ord_idx, obs_i) in enumerate(zip(flagged_orders, flagged_obs)):
            print(f"{i}: Order {ord_idx}, Observation {obs_i}")
        print("=" * 40)
        
    # If no indices provided, use the first one
    if order_idx is None or obs_idx is None:
        order_idx = flagged_orders[0]
        obs_idx = flagged_obs[0]
        print(f"\nPlotting first flagged: Order {order_idx}, Observation {obs_idx}\n")
    
    # Get previous, current, next observation indices
    n_obs = spec_arr.shape[2]
    prev_obs = max(0, obs_idx - 1)
    next_obs = min(n_obs - 1, obs_idx + 1)
    
    # Extract spectra
    prev_spec = spec_arr[order_idx, :, prev_obs]
    curr_spec = spec_arr[order_idx, :, obs_idx]
    next_spec = spec_arr[order_idx, :, next_obs]
    
    wave = wave_arr[order_idx, :, obs_idx]
    
    # Plot with colorblind-friendly palette
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10.colors  # colorblind-friendly tableau10
    
    ax.plot(wave, prev_spec, alpha=0.75, linewidth=1, 
            label=f'{date_arr[prev_obs]}')
    ax.plot(wave, curr_spec, alpha=1.0, linewidth=1, 
            label=f'{date_arr[obs_idx]} [FLAGGED]')
    ax.plot(wave, next_spec, alpha=0.75, linewidth=1, 
            label=f'{date_arr[next_obs]}')
    
    ax.set_xlabel('Angstroms')
    ax.set_ylabel('Relative Flux')
    ax.set_title(f'Order {order_idx}, Observation {obs_idx}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    return fig, ax