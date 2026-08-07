import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import Voigt1D
from lmfit.models import VoigtModel
# from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm

from load_data import debug_print 
from figures import plot_laser

np.random.seed(seed=42)

# ___LASER DETECTION___

# formulas
def mse(obs, pred):
    return np.sum((obs - pred)**2) / len(obs)
    
def med_abs_dev(x): # 2 for obs
    med = np.nanmedian(x)
    abs_dev = np.abs(x - med)
    mad = np.nanmedian(abs_dev)
    return mad

def fwhm_voigt_to_gauss(f_V, method="kol"):
    """
    Assumes FWHM same for Lorentz and Gauss
    Kielkopf and Olivero and Longbothum 
    """
    if method == "whiting":
        f_L = f_V / (0.5 + np.sqrt(5/4))
    if method == "kol":
        f_L = f_V / (0.5343 + np.sqrt(1.2169)) 
    return f_L

def get_residual(spec_arr):
    """Subtract median spectrum from each observation"""
    median_obs = np.nanmedian(spec_arr, axis=2)  # Shape: (ords, wave_cols)
    
    # Broadcasting: (ords, wave_cols, obs) - (ords, wave_cols, 1)
    residual_arr = spec_arr - median_obs[:, :, np.newaxis]
    
    return residual_arr

# fwhm equations
def distance(x1, y1, x2, y2):
    xdiff = x1-x2
    ydiff = y1-y2
    return np.sqrt(xdiff**2 + ydiff**2)

def find_close_points(x, y, x1, y1):
    x1_idx = np.argmin(np.abs(x - x1))

    # Create mask for indices within ±10 of x1
    nearby_mask = (np.arange(len(x)) >= x1_idx - 10) & (np.arange(len(x)) <= x1_idx + 10)
    
    tl = ((x < x1) & (y > y1) & nearby_mask)
    tr = ((x > x1) & (y > y1) & nearby_mask)
    bl = ((x < x1) & (y < y1) & nearby_mask)
    br = ((x > x1) & (y < y1) & nearby_mask)
    lims = (tl, bl, tr, br)
    
    close_idxs = [0] * 4
    for i, idx in enumerate(lims):

        dist_arr = distance(x1, y1, x[idx], y[idx])
        try:
            close_idxs[i] = np.argmin(dist_arr)
        except ValueError as e:
            close_idxs[i] = np.nan

    return close_idxs, lims

def get_a_fwhm(x, y, x1, y1, peakidx, verbose):
    close_idxs, lims = find_close_points(x, y, x1, y1)

    close_xs = [0] * 4
    for i, idx in enumerate(lims):
        if np.isnan(close_idxs[i]):
            close_xs[i] = x[peakidx]
        else:
            close_xs[i] = x[idx][close_idxs[i]]
    low_x = np.abs(close_xs[0] - close_xs[1]) + close_xs[1]
    high_x = np.abs(close_xs[2] - close_xs[3]) + close_xs[2]
    fwhm = high_x - low_x

    debug_print(verbose, f"low_x = {low_x}, high_x = {high_x}")
    return fwhm, low_x, high_x, close_idxs, lims

# post dave convo -- this will likely identify edges but thats ok
def full_width_half_max(x, y, peaks, half_maxes, verbose=False, plot=False):
    fwhm_arr = np.empty_like(peaks, dtype=float)
    
    peakx = x[peaks]
    
    for i in range(len(peaks)):
        
        debug_print(verbose, "center freq", peakx[i])
        
        fwhm, low_x, high_x, close_idxs, lims = get_a_fwhm(x, y, peakx[i], half_maxes[i], peaks[i], verbose=verbose)
        
        fwhm_arr[i] = fwhm
        
        if plot:
            plot_laser(x, y, peaks[i], peakx[i], half_maxes[i], close_idxs, lims, low_x, high_x, xlim=[low_x, high_x])
            plt.show()
    return fwhm_arr

def wave_to_fwhms(wave, flux, sigma, poly, 
                  residual, coeff, 
                  max_diff=0.01, 
                  threshold_type="mad", interp_samples=None,
                  verbose=False, plot=False):

    if interp_samples is not None:
        interp_wave = np.linspace(wave[0], wave[-1], num=interp_samples)
        flux = np.interp(interp_wave, wave, flux)
        sigma = np.interp(interp_wave, wave, sigma)
        poly = np.interp(interp_wave, wave, poly)
        residual = np.interp(interp_wave, wave, residual)

        wave = interp_wave
        
    if threshold_type == "std":
        threshold = poly + coeff * np.sqrt(sigma**2 + np.nanstd(residual)**2)
    elif threshold_type == "mad":
        threshold = poly + coeff * np.sqrt(sigma**2 + med_abs_dev(residual)**2) 

    peaks, _ = find_peaks(flux, threshold) # get peaks above threshold 
    debug_print(verbose, f"scipy peaks: {len(peaks)}")
    wave_pks, flx_pks, poly_pks = wave[peaks], flux[peaks], poly[peaks]

    half_maxes = poly_pks + 0.5*(flx_pks - poly_pks)

    # pre dave
    # fwhms, x_peaks, valid_mask = full_width_half_max(wave, flux, 
    #                                      wave_pks, half_maxes, 
    #                                      max_diff, verbose=verbose) # fwhm of peaks

    # post dave
    fwhms = full_width_half_max(wave, flux, peaks, half_maxes, verbose=verbose, plot=plot) # fwhm of peaks

    debug_print(verbose, f"fwhm: {fwhms}")
    
    return (fwhms, wave_pks, half_maxes, 
            flx_pks, threshold, 
            wave, flux, poly, residual)

def check_windex(wl):
    if wl <= 6000:
        windex = 0
    elif (wl > 6000) and (wl <= 9600):
        windex = 1
    elif wl > 9600:
        windex = 2

    return windex


def lsf_per_wav(wave, wl,
                amplitude_L=1, 
                w_lorentz=np.array([0.28, 0.21, 0.17]) * 1e-5, 
                w_gauss=np.array([1.0, 1.01, 1.18]) * 1e-5,
                broaden_coeff=1,
                set_fwhm=None,
               model_type="astropy"):
    
    windex = check_windex(wl)

    if set_fwhm is not None:
        gauss_fwhm = fwhm_voigt_to_gauss(set_fwhm)
        fwhm_G = gauss_fwhm
        fwhm_L = gauss_fwhm
    else: 
        fwhm_G = w_gauss[windex]*wl
        fwhm_L = w_lorentz[windex]*wl
        
    if model_type == "astropy":
        v1 = Voigt1D(x_0=wl, 
                     amplitude_L=amplitude_L, 
                     fwhm_L=fwhm_L*broaden_coeff, 
                     fwhm_G=fwhm_G*broaden_coeff)

        return v1(wave)
        
    if model_type == "lmfit":
        model = VoigtModel()
        v1 = model.eval(amplitude=amplitude_L, 
                      center=wl, 
                      sigma=fwhm_G*broaden_coeff / 2.35482004503, 
                      gamma=fwhm_L*broaden_coeff / 2, 
                      x=wave)
        return v1

# Find which orders contain this wavelength
def identify_orders(new_wave_arr, wl):
    mins = np.nanmin(new_wave_arr, axis=(1, 2))
    maxs = np.nanmax(new_wave_arr, axis=(1, 2))
    orders = np.where((mins <= wl) & (maxs >= wl))[0]
    return orders 
        
# Find column index closest to target wavelength in this order/obs
def get_colidx(new_wave_arr, ordidx, obsidx, wl):
    wave = new_wave_arr[ordidx, :, obsidx]
    col_idx = np.nanargmin(np.abs(wave - wl))  # Closest column
    return wave, col_idx

def comb_sig_per_wl(wave_arr, wl, obsidx,
                    combined_sigma_ords):
    
    orders = identify_orders(wave_arr, wl)

    sigmas = np.empty_like(orders, dtype=float)
    for i, ordidx in enumerate(orders): 
        _, col_idx = get_colidx(wave_arr, 
                             ordidx, 
                             obsidx, wl)
        sigma = combined_sigma_ords[ordidx][col_idx]
        sigmas[i] = sigma

    return sigmas

def load_combined_sigma_dict(dir_path, n_obs):
    combined_sigma_dict = {}
    for obsidx in range(n_obs):
        combined_sigma_dict[obsidx] = np.load(
            dir_path + f"/combined_sigma/combined_sigma_{obsidx}.npy"
        )
    return combined_sigma_dict

def compute_wl_to_px_grid(new_wave_arr, n_ords, n_obs):

    wl_to_px_grids = {}
    for order in range(n_ords):
        for obsidx in range(n_obs):
            wave = new_wave_arr[order, :, obsidx]
            pixels = np.linspace(0, len(wave), len(wave))
            # Only compute where wave is not NaN
            valid = ~np.isnan(wave)
            if np.any(valid):
                # Polyfit only once per order/observation
                wave_of_px = np.polyfit(pixels[valid], wave[valid], 1)
                wl_to_px_grids[(order, obsidx)] = wave_of_px
                
    return wl_to_px_grids

def get_wl_to_orders(new_wave_arr, wls, verbose):
    wl_to_orders = {}
    mins = np.nanmin(new_wave_arr, axis=(1, 2))
    maxs = np.nanmax(new_wave_arr, axis=(1, 2))
    
    for w, wl in enumerate(wls):
        orders = np.where((mins <= wl) & (maxs >= wl))[0]
        wl_to_orders[wl] = orders
        debug_print(verbose, f"WL {wl}: orders {orders}")

    return wl_to_orders

def get_laser_params(dir_path, new_wave_arr, wls, verbose=False):
    n_ords, _, n_obs = new_wave_arr.shape
    combined_sigma_dict = load_combined_sigma_dict(dir_path, n_obs)
    wl_to_px_grids = compute_wl_to_px_grid(new_wave_arr, n_ords, n_obs)
    wl_to_orders = get_wl_to_orders(new_wave_arr, wls, verbose)

    return combined_sigma_dict, wl_to_px_grids, wl_to_orders

# Claude optimized
def make_laser_arr(dir_path, 
                   wls, 
                   combined_sigma_dict, 
                   wl_to_px_grids,
                   wl_to_orders,
                   mult=1.5, 
                   n=3, 
                   broaden_coeff=1,
                   model_type="astropy",
                   set_fwhm_px=None,
                   verbose=False):
    
    results = np.load(dir_path + "/results.npz")
    new_wave_arr = results['new_wave_arr']
    normalized_spec = results['normalized_spec']
    poly_arr_best = results['poly_arr_best']
    
    n_ords, n_cols, n_obs = new_wave_arr.shape
    laser_arr = np.zeros((n_ords, n_cols, n_obs))
    
    # Main loop: iterate by observation (keeps file in cache)
    for obsidx in range(n_obs):
        combined_sigma_ords = combined_sigma_dict[obsidx]
        
        for wl in wls:
            orders = wl_to_orders[wl]
            
            # OPTIMIZATION 4: Get all col_idx for this wl/obsidx at once
            col_indices = {}
            for order in orders:
                wave = new_wave_arr[order, :, obsidx]
                col_idx = np.nanargmin(np.abs(wave - wl))
                col_indices[order] = (wave, col_idx)
            
            # OPTIMIZATION 5: Batch Voigt1D evaluation
            # Instead of creating new models, evaluate all at once
            for i, order in enumerate(orders):
                wave, col_idx = col_indices[order]
                
                # Get sigma
                sigma = combined_sigma_ords[order][col_idx]
                amplitude = np.abs(poly_arr_best[order, col_idx, obsidx] 
                                   * sigma * mult)
                
                # Compute FWHM once
                if set_fwhm_px is not None:
                    pixels = np.linspace(0, len(wave), len(wave))
                    wave_of_px = np.polyfit(pixels, wave, 1)
                    g = np.poly1d(wave_of_px)

                    half_width = set_fwhm_px / 2

                    set_fwhm = g(col_idx + half_width) - g(col_idx - half_width)
                                    
                # Evaluate LSF
                lsf_vals = lsf_per_wav(wave, 
                                       wl,
                                       amplitude_L=amplitude,
                                       broaden_coeff=broaden_coeff,
                                       set_fwhm=set_fwhm,
                                       model_type=model_type)
                
                laser_arr[order, :, obsidx] += lsf_vals
    
    return laser_arr
    

def fwhm_test(wave, x_peaks, method="pixel", px_min=1, px_max=4, amplitude_L=1, broaden_coeff=1, model_type="astropy"):
    
    if method == "pixel":
        # pixel - wavelength function to convert fwhm to pixels
        pixels = np.arange(0, len(wave), 1)
        fit_coeffs = np.polyfit(pixels, wave, 1)
        wave_of_px = np.poly1d(fit_coeffs)
        
        fwhm_min = wave_of_px(px_min) - wave_of_px(0) # with flux conserving resampler, wave grids are ~linear
        fwhm_max = wave_of_px(px_max) - wave_of_px(0)
        return fwhm_min, fwhm_max

    if method == "model":
        unfilled_ranges = [True, True, True]
        lsfs = np.ones((3, wave.shape[0]))
        fwhm_lsfs = np.empty(3) 
    
        for wl in x_peaks:
        
            windex = check_windex(wl)
        
            if unfilled_ranges[windex]:
                
                lsfs[windex, :] = lsf_per_wav(wave, wl,
                            amplitude_L=amplitude_L, broaden_coeff=broaden_coeff)
            
                fwhm, _, _, _, _, _, _, _, _ = wave_to_fwhms(wave, lsfs[windex, :], 
                                               np.zeros_like(wave), 
                                               np.zeros_like(wave), 
                                               np.zeros_like(wave), 
                                               0, max_diff=0.5)
            
                fwhm_lsfs[windex] = fwhm[0]
                
                unfilled_ranges[windex] = False
        
            else:
                continue

        return fwhm_lsfs


def thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose, plot=False):
    
    (fwhms, x_peaks, half_maxes, 
    flx_pks, threshold, 
    wave, flux, poly, residual) = wave_to_fwhms(wave, 
                                           flux, 
                                           sigma,
                                           poly,
                                           residual,
                                           coeff, 
                                           max_diff=max_diff, 
                                           threshold_type=threshold_type, 
                                           interp_samples=interp_samples, 
                                           verbose=verbose, plot=plot)
    
    min_lsf_fwhms, max_lsf_fwhms = fwhm_test(wave, x_peaks, method=method, px_min=px_min)
    
    fwhm_test_pass = fwhms[(fwhms > min_lsf_fwhms) & (fwhms < max_lsf_fwhms)] 
    # doesnt work with method"model"
    x_test_pass = x_peaks[(fwhms > min_lsf_fwhms) & (fwhms < max_lsf_fwhms)]
    
    return (fwhms, x_peaks, 
            half_maxes, flx_pks, 
            threshold, 
            wave, flux, 
            poly, residual, 
            min_lsf_fwhms, max_lsf_fwhms,
            fwhm_test_pass, 
            x_test_pass)


# default args above
# 1-10s runtime
def get_fwhm_arr(new_wave_arr, flux_arr, 
                 poly_arr_best, residual_arr, 
                 normalized_sig, coeff, **kwargs):

    max_diff = kwargs.get('max_diff', 0.01)
    threshold_type = kwargs.get('threshold_type', 'mad')
    interp_samples = kwargs.get('interp_samples', 50000)
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', "pixel")
    px_min = kwargs.get('px_min', 2.5)
    save_dir = kwargs.get('save_dir', None)
    plot=kwargs.get('plot', False)
    
    norders = new_wave_arr.shape[0]
    nobs = new_wave_arr.shape[2]
    
    fwhm_arr = np.empty((norders, nobs), dtype=object)
    
    for ordidx in range(norders):
        for obsidx in range(nobs):
            wave = new_wave_arr[ordidx, :, obsidx]
            flux = flux_arr[ordidx, :, obsidx]
            poly = poly_arr_best[ordidx, :, obsidx]
            residual = residual_arr[ordidx, :, obsidx]
            sigma = normalized_sig[ordidx, :, obsidx]
            
            (fwhms, x_peaks, 
            half_maxes, flx_pks, 
            threshold, 
            wave, flux, 
            poly, residual, 
            min_lsf_fwhms, max_lsf_fwhms,
            fwhm_test_pass, 
            x_test_pass) = thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose, plot=plot)
            
            # Store as dictionary
            fwhm_arr[ordidx, obsidx] = {
                'fwhms': fwhms,
                'x_peaks': x_peaks,
                'half_maxes': half_maxes,
                'flx_pks': flx_pks,
                'threshold': threshold,
                'wave': wave, 
                'flux':flux,
                'poly': poly,
                'residual': residual, 
                'min_lsf_fwhms': min_lsf_fwhms,
                'max_lsf_fwhms': max_lsf_fwhms,
                'fwhm_test_pass': fwhm_test_pass,
                'x_test_pass': x_test_pass
            }

    if save_dir:
        np.savez(savedir + "/peaks_arr.npz", fwhm_arr)

    return fwhm_arr
    
def save_fwhm_per_obs(dir_path,
                      wave_arr, flux_arr, 
                      sigma_arr, poly_arr, 
                      residual_arr, mult=None, 
                      wls=None, coeff=1, 
                      max_diff=0.01, 
                      threshold_type='mad', 
                      interp_samples=None, 
                      method='pixel', 
                      px_min=2.5, 
                      verbose=False, 
                      all_data=False, 
                      save_folder=False,
                     plot=False):
        
    norders = wave_arr.shape[0]
    nobs = wave_arr.shape[2]

    alph_arr = np.empty((nobs), dtype=object)
    for obsidx in range(nobs): # per observation
        obs_arr = np.empty((norders), dtype=object)
        debug_print(verbose, f"Processing Obs {obsidx}")

        for ordidx in range(norders): # per order
            wave = wave_arr[ordidx, :, obsidx]
            flux = flux_arr[ordidx, :, obsidx]
            sigma = sigma_arr[ordidx, :, obsidx]
            poly = poly_arr[ordidx, :, obsidx]
            residual = residual_arr[ordidx, :, obsidx]
            
            (fwhms, x_peaks, 
            half_maxes, flx_pks, 
            threshold, 
            wave, flux, 
            poly, residual, 
            min_lsf_fwhms, max_lsf_fwhms,
            fwhm_test_pass, 
            x_test_pass) = thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose, plot=plot)
            
            if all_data:
                obs_arr[ordidx] = {
                    'fwhms': fwhms,
                    'x_peaks': x_peaks,
                    'half_maxes': half_maxes,
                    'flx_pks': flx_pks,
                    'threshold': threshold,
                    'wave': wave, 
                    'flux':flux,
                    'poly': poly,
                    'residual': residual, 
                    'min_lsf_fwhms': min_lsf_fwhms,
                    'max_lsf_fwhms': max_lsf_fwhms,
                    'fwhm_test_pass': fwhm_test_pass,
                    'x_test_pass': x_test_pass,
                    'mult': mult, 
                    'wls': wls
                }
            else:
                obs_arr[ordidx] = {

                    'x_test_pass': x_test_pass,
                    'mult': mult, 
                    'wls': wls
                }

        if save_folder: 
            save_path = dir_path + save_folder # e.g. "/base_peaks"
            if os.path.exists(save_path) is False: 
                os.mkdir(save_path)
    
            save_file = save_path + f"{save_folder}_{obsidx}.npz"
            np.savez(save_file, obs_arr)
    
            debug_print(verbose, ("saved to:", save_file))

        else: 
            alph_arr[obsidx] = obs_arr

    return alph_arr

def generate_inj_params(low=5140, high=10400, 
                        length=100, buffer=10, 
                        magic_wls=None):
    # from hippke 2018
    if magic_wls is None:
        magic_wls = [5321, 6565, 5891] # ang
    
    wls = np.random.uniform(low=low, high=high, size=(length,))

    # Vectorized pairwise distance check
    distances = np.abs(wls[:, np.newaxis] - wls[np.newaxis, :])
    
    # Get indices of upper triangle (excludes diagonal, avoids duplicates)
    i, j = np.triu_indices_from(distances, k=1)
    
    # Find pairs within buffer
    mask = distances[i, j] < buffer
    violations = list(zip(i[mask], j[mask]))
    
    if violations:
        # Move one wavelength from each violation
        for idx1, idx2 in violations:
            
            for _ in range(100):
                new_wl = np.random.uniform(low, high)
                
                # Vectorized check: is new_wl valid?
                min_distance = np.min(np.abs(new_wl - np.delete(wls, idx2)))
                if min_distance >= buffer:
                    wls[idx2] = new_wl
                    break
    
    for magic_wl in magic_wls:
        closest_idx = np.argmin(np.abs(wls - magic_wl))
        wls[closest_idx] = magic_wl

    return wls

def get_ord_to_wl_arr(wave_arr, wls):
    nords = wave_arr.shape[0]

    # Change this line:
    order_to_wls = {i: [] for i in range(nords)}
    
    for wl in wls:
        order_list = identify_orders(wave_arr, wl)
        for order in order_list:
            if order not in order_to_wls:
                order_to_wls[order] = []
            order_to_wls[order].append(wl)
        
    order_to_wls_arr = np.array([order_to_wls[order] for order in range(nords)], 
                                dtype=object)

    return order_to_wls_arr

from scipy.optimize import linear_sum_assignment

def hungarian_bipartite(wls_arr, peaks_arr, tolerance=0.05):
    # Create cost matrix
    cost_matrix = np.abs(wls_arr[:, np.newaxis] - peaks_arr[np.newaxis, :])
    
    # Set costs to infinity for pairs exceeding tolerance
    cost_matrix[cost_matrix > tolerance] = np.inf
    
    # Initialize result
    recovered_wls = np.zeros(len(wls_arr), dtype=bool)
    
    try:
        wl_indices, peak_indices = linear_sum_assignment(cost_matrix)
        
        # Mark only valid matches as True
        for wl_idx, peak_idx in zip(wl_indices, peak_indices):
            if np.isfinite(cost_matrix[wl_idx, peak_idx]):
                recovered_wls[wl_idx] = True
                
    except ValueError:
        # If infeasible, all wavelengths remain False
        pass

    return recovered_wls

def per_alpha_recovery(wave_arr, inj_list, tolerance=0.05):
    nords, ncols, nobs = wave_arr.shape

    obs_recovered = np.empty((nords, nobs), dtype=object)
    for obsidx, inj_path in enumerate(inj_list):
        injection = np.load(inj_path, allow_pickle=True)
        inj_arr = injection['arr_0'] 
    
        if obsidx == 0: 
            ordidx = 0
            wls = np.sort(inj_arr[ordidx]['wls']) # wls arr same for all ords
    
            mult = inj_arr[ordidx]['mult'] #mult same for all obs (1 mult per alpha)
            
            order_to_wls_arr = get_ord_to_wl_arr(wave_arr, wls)
    
        for ordidx in range(nords):
            x_test_pass = inj_arr[ordidx]['x_test_pass']
            wls_in_order = order_to_wls_arr[ordidx]
            wls_arr = np.array(wls_in_order)
            peaks_arr = np.array(x_test_pass)

            recovered_wls = hungarian_bipartite(wls_arr, 
                                                peaks_arr, 
                                                tolerance=tolerance)
    
            #     obs_wls.append(wls_in_order)
            #     obs_mult.append(mult)
            obs_recovered[ordidx, obsidx] = recovered_wls

    return order_to_wls_arr, mult, obs_recovered

def per_alpha_recovery_arr(wave_arr, one_alph, tolerance=0.05):
    nords, ncols, nobs = wave_arr.shape

    obs_recovered = np.empty((nords, nobs), dtype=object)
    for obsidx in range(nobs):
    
        if obsidx == 0: 
            ordidx = 0
            wls = np.sort(one_alph[obsidx][ordidx]['wls']) # wls arr same for all ords
    
            mult = one_alph[obsidx][ordidx]['mult'] #mult same for all obs (1 mult per alpha)
            
            order_to_wls_arr = get_ord_to_wl_arr(wave_arr, wls)
    
        for ordidx in range(nords):
            x_test_pass = one_alph[obsidx][ordidx]['x_test_pass']
            wls_in_order = order_to_wls_arr[ordidx]
            wls_arr = np.array(wls_in_order)
            peaks_arr = np.array(x_test_pass)

            recovered_wls = hungarian_bipartite(wls_arr, 
                                                peaks_arr, 
                                                tolerance=tolerance)
    
            #     obs_wls.append(wls_in_order)
            #     obs_mult.append(mult)
            obs_recovered[ordidx, obsidx] = recovered_wls

    return order_to_wls_arr, mult, obs_recovered

def recovery_1d(all_alph, wave_arr, tolerance=0.05):

    all_wls_recovered, all_wls_not_recovered = [], []
    all_mult_recovered, all_mult_not_recovered = [], []
    nords, ncols, nobs = wave_arr.shape
    
    for aphidx in tqdm(range(len(all_alph))):
        order_to_wls_arr, mult, obs_recovered = per_alpha_recovery_arr(wave_arr, 
                                                                       all_alph[aphidx], 
                                                                       tolerance=tolerance)
        for obsidx in range(nobs):
            for ordidx in range(nords):
                wls = np.array(order_to_wls_arr[ordidx])
                recovered = obs_recovered[ordidx, obsidx]
                
                # Skip if no data for this order-obs pair
                if recovered is None:
                    continue
                
                # Separate by recovery status
                wls_recovered = wls[recovered]
                wls_not_recovered = wls[~recovered]
    
                mult_recovered = np.zeros_like(wls_recovered) + mult
                mult_not_recovered = np.zeros_like(wls_not_recovered) + mult
    
                all_wls_recovered.append(wls_recovered)
                all_wls_not_recovered.append(wls_not_recovered)
                all_mult_recovered.append(mult_recovered)
                all_mult_not_recovered.append(mult_not_recovered)

    plt_wls_recovered = np.hstack(all_wls_recovered) # plt meaning for plotting, aka 1d
    plt_wls_not_recovered = np.hstack(all_wls_not_recovered)
    plt_mult_recovered = np.hstack(all_mult_recovered)
    plt_mult_not_recovered = np.hstack(all_mult_not_recovered)

    return plt_wls_recovered, plt_wls_not_recovered, plt_mult_recovered, plt_mult_not_recovered

def get_rr(alphas, plt_mult_recovered, plt_mult_not_recovered, 
           plt_wls_recovered, plt_wls_not_recovered, wl_filter_range=None):
    rr = np.empty((len(alphas)))
    for i, alpha in enumerate(alphas):
        reidxs = np.where(plt_mult_recovered == alpha)[0]
        nridxs = np.where(plt_mult_not_recovered == alpha)[0]
        wl_rec = plt_wls_recovered[reidxs]
        wl_not_rec = plt_wls_not_recovered[nridxs]
    
        if wl_filter_range is not None:
            wfrmin, wfrmax = wl_filter_range[0], wl_filter_range[1]
            wl_rec_mask = np.where((wl_rec > wfrmin) & (wl_rec < wfrmax))
            wl_not_rec_mask = np.where((wl_not_rec > wfrmin) & (wl_not_rec < wfrmax))
            wl_rec = wl_rec[wl_rec_mask]
            wl_not_rec = wl_not_rec[wl_not_rec_mask]
        n_rec = len(wl_rec)
        n_not_rec = len(wl_not_rec)
    
        try: 
            rr[i] = n_rec / (n_rec + n_not_rec)
        except ZeroDivisionError as e:
            rr[i] = 0
            
    return rr

def get_ir_threshold(plt_wls_recovered, plt_wls_not_recovered, 
                     plt_mult_recovered, plt_mult_not_recovered, rr_thresh=0.997):
    unique_wls_recovered = np.unique(plt_wls_recovered)
    # min_alpha_recovered_vals = np.array([plt_mult_recovered[plt_wls_recovered == w].min() 
    #                                      for w in unique_wls_recovered])
    
    # Not recovered case
    unique_wls_not_recovered = np.unique(plt_wls_not_recovered)
    # min_alpha_not_recovered_vals = np.array([plt_mult_not_recovered[plt_wls_not_recovered == w].min() 
    #                                          for w in unique_wls_not_recovered])
    
    recovered_pairs = np.column_stack((plt_wls_recovered, plt_mult_recovered))
    unique_pairs = np.unique(recovered_pairs, axis=0)
    
    min_alphas = []
    filtered_wls = []
    
    for wls, mult in unique_pairs:
        # Count this pair in recovered
        count_rec = np.sum((plt_wls_recovered == wls) & (plt_mult_recovered == mult))
        # Count this pair in not_recovered
        count_not_rec = np.sum((plt_wls_not_recovered == wls) & (plt_mult_not_recovered == mult))
        
        # Calculate recovery rate
        recovery_rate = count_rec / (count_rec + count_not_rec)
        
        if recovery_rate > rr_thresh:
            # Check if we've already seen this wavelength
            if wls in filtered_wls:
                # Update with minimum mult
                idx = filtered_wls.index(wls)
                if mult < min_alphas[idx]:
                    min_alphas[idx] = mult
            else:
                # First time seeing this wavelength
                filtered_wls.append(wls)
                min_alphas.append(mult)
    
    min_alphas = np.array(min_alphas)
    filtered_wls = np.array(filtered_wls)

    return min_alphas, filtered_wls