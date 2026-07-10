import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import cycle

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
        
def plot_spectra_elike(fig, axs, x, y, n_sections, title=None, xlabel=None, ylabel=None):
    x_sections = np.array_split(x, n_sections)
    y_sections = np.array_split(y, n_sections)

    axs[-1].set(xlabel=xlabel if xlabel is not None else None)
    axs[(n_sections//2)].set(ylabel=ylabel if ylabel is not None else None)
    fig.suptitle(title if title is not None else None)

    for i in range(n_sections):
        axs[i].plot(x_sections[i], y_sections[i])

def plot_spectra_ords(wave_arr, data, obs_idx=0, data2=None, figsize=(36, 20), data2_label="Best Fit"):
    """
    Plot all 60 orders for a single observation in a 6x10 grid.
    
    Parameters:
    -----------
    wave_arr : ndarray, shape (orders, wave_cols, observations)
        Wavelength array
    data : ndarray, shape (orders, wave_cols, observations)
        Spectrum data to plot
    obs_idx : int
        Observation index to plot
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(6, 10, figsize=figsize)
    axes = axes.flatten()  # Flatten to 1D for easy iteration
    
    for order in range(60):
        ax = axes[order]
        
        wave = wave_arr[order, :, obs_idx]
        spec = data[order, :, obs_idx]
        
        ax.plot(wave, spec, label=f'Order {order}')

        if data2 is not None:
            poly = data2[order, :, obs_idx]
            ax.plot(wave, poly, label=data2_label, zorder=5, linewidth=2)
        # Axis labels only on outer edges
        if order >= 50:  # Bottom row
            ax.set_xlabel('Wavelength', fontsize=9)
        if order % 10 == 0:  # Leftmost column
            ax.set_ylabel('Flux', fontsize=9)
        
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    return fig, axes

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

def plot_high_std_compare(spec_arr, wave_arr, date_arr, high_std_mask, order_idx=None, obs_idx=None, verbose=False):
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

def plot_bin_polyfit(new_wave_arr, 
                     normalized_spec, 
                     poly_arr_best, 
                     bin_midpts_arr_best, 
                     bin_meds_arr_best, 
                     ordidx=0, obsidx=0):
    
    fig = plt.figure(figsize=(16, 6))
    plt.plot(new_wave_arr[ordidx, :, obsidx], 
             normalized_spec[ordidx, :, obsidx], 
             label='raw data')
    plt.plot(new_wave_arr[ordidx, :, obsidx], 
             poly_arr_best[ordidx, :, obsidx], 
             label="bin polyfit", 
             linewidth=4)
    plt.scatter(bin_midpts_arr_best[ordidx, :, obsidx], 
                bin_meds_arr_best[ordidx, :, obsidx], 
                label="bins", s=50, color="red", zorder=10)
    
    plt.xlabel("Angstroms")
    plt.ylabel("Relative Flux")
    plt.legend()
    
    return fig 

def plot_bic_deg(crit_vals, deg_vals, criterion="BIC"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # crit values
    im1 = axes[0].imshow(crit_vals, aspect='auto', norm=mpl.colors.LogNorm(vmax=np.percentile(crit_vals, 98)))
    axes[0].set_title(f'{criterion} Values', fontsize=12)
    axes[0].set_xlabel('Observations')
    axes[0].set_ylabel('Orders')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label(criterion)
    
    # Degree values
    deg_vals = deg_vals.astype(int)
    cmap = plt.get_cmap('viridis', np.max(deg_vals) - np.min(deg_vals) + 1)
    im2 = axes[1].imshow(deg_vals, cmap=cmap, aspect='auto',
                         vmin=np.min(deg_vals) - 0.5, 
                         vmax=np.max(deg_vals) + 0.5)
    axes[1].set_title('Best Polynomial Degree', fontsize=12)
    axes[1].set_xlabel('Observations')
    axes[1].set_ylabel('Orders')
    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=np.arange(np.min(deg_vals), np.max(deg_vals) + 1))
    cbar2.set_label('Degree')
    
    plt.tight_layout()

    return fig, axes

## ___LASER PLOTS ___
def plot_ordobs_peaks(fwhm_arr, 
                      ordidx = 0, obsidx = 0, figsize=(20, 6), 
                      style=tableau_cb10):
    colors = cycle(style)
    
    peaks_data = fwhm_arr[ordidx, obsidx]
    
    plt.figure(figsize=figsize)
    plt.plot(peaks_data['wave'], peaks_data['flux'], label="Spectrum", color=next(colors))
    plt.scatter(peaks_data['x_peaks'], peaks_data['flx_pks'], 
                label="Peaks", color=next(colors))
    plt.plot(peaks_data['x_peaks'], peaks_data['half_maxes'], 
                label="Half-width", color=next(colors), linestyle="dashed")
    plt.plot(peaks_data['wave'], peaks_data['threshold'], 
                label="Threshold", color=next(colors), linestyle="dashed")
    plt.legend()
    plt.grid()

def plot_spectra_ords_peaks(fwhm_arr, obs_idx=0, figsize=(36, 20), style=tableau_cb10):
    """
    Plot all 60 orders for a single observation in a 6x10 grid.
    Alongside detected peaks
    """
    fig, axes = plt.subplots(6, 10, figsize=figsize)
    axes = axes.flatten()  # Flatten to 1D for easy iteration
    
    for order in range(60):
        colors = cycle(style)
        
        ax = axes[order]

        peaks_data = fwhm_arr[order, obs_idx]
        
        wave = peaks_data['wave']
        spec = peaks_data['flux']
        x_peaks = peaks_data['x_peaks']
        flx_pks = peaks_data['flx_pks']
        half_maxes = peaks_data['half_maxes']
        threshold = peaks_data['threshold']
        
        ax.plot(wave, spec, label=f'Order {order}', color=next(colors))
        ax.scatter(x_peaks, flx_pks, 
                label="Peaks", color=next(colors))
        ax.plot(x_peaks, half_maxes, 
                    label="Half-width", color=next(colors), linestyle="dashed")
        ax.plot(wave, threshold, 
                    label="Threshold", color=next(colors), linestyle="dashed")
        
        # Axis labels only on outer edges
        if order >= 50:  # Bottom row
            ax.set_xlabel('Wavelength', fontsize=9)
        if order % 10 == 0:  # Leftmost column
            ax.set_ylabel('Flux', fontsize=9)
        
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    return fig, axes

def plot_recovery_rate(tolerances, recovered_percentage, recovered_percentage_pass):
    plt.plot(tolerances, recovered_percentage, label="All Injected Peaks")
    plt.plot(tolerances, recovered_percentage_pass, label="> LSF FWHM")
    plt.xscale("log")
    plt.grid()
    plt.ylabel("% Recovered Injections")
    plt.xlabel("Allowed $\AA$ Away from Injection")
    plt.legend()