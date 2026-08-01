import os

import numpy as np

from astropy.modeling.models import Voigt1D
from lmfit.models import VoigtModel
# from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm

from load_data import debug_print 

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
    tl = ((x < x1) & (y > y1))
    tr = ((x > x1) & (y > y1))
    bl = ((x < x1) & (y < y1))
    br = ((x > x1) & (y < y1))
    lims = (tl, bl, tr, br)
    
    close_idxs = [0] * 4
    for i, idx in enumerate(lims):

        dist_arr = distance(x1, y1, x[idx], y[idx])
        try:
            close_idxs[i] = np.argmin(dist_arr)
        except ValueError as e:
            close_idxs[i] = np.nan

    return close_idxs, lims

def get_a_fwhm(x, y, x1, y1, peakidx):
    close_idxs, lims = find_close_points(x, y, x1, y1)

    close_xs = [0] * 4
    for i, idx in enumerate(lims):
        if np.isnan(close_idxs[i]):
            close_xs[i] = x[peakidx]
        else:
            close_xs[i] = x[idx][close_idxs[i]]
    low_x = np.abs(close_xs[0] - close_xs[1])
    high_x = np.abs(close_xs[2] - close_xs[3])
    fwhm = high_x + low_x

    return fwhm, low_x, high_x

# pre dave convo
# def full_width_half_max(x, y, peakx, half_maxes, max_diff=0.01, verbose=False):
#     fwhm_arr = []
#     x_peaks = []
#     valid_mask = np.zeros(len(half_maxes), dtype=bool)  # True = peak passed, False = excluded
    
#     for i, half_max in enumerate(half_maxes):
#         # half_max = peak/2
#         center_freq = peakx[i]

#         debug_print(verbose, "center freq", center_freq)

#         x_args = np.where(np.abs(y - half_max) < max_diff)
#         x_vals = x[x_args]

#         debug_print(verbose, 
#                     f"{len(x_vals)} intersections between the spectrum and half max of this peak.")

#         lower_mask = (x_vals < center_freq)
#         upper_mask = (x_vals > center_freq)

#         if np.any(lower_mask == True) and np.any(upper_mask == True):
#             lower_freq_arg = np.argmin(np.abs(center_freq - x_vals[lower_mask]))
#             upper_freq_arg = np.argmin(np.abs(center_freq - x_vals[upper_mask]))
    
#             lower_freq = x_vals[lower_mask][lower_freq_arg]
#             upper_freq = x_vals[upper_mask][upper_freq_arg]
#             debug_print(verbose, "lower, upper freqs", lower_freq, upper_freq)
#             fwhm = upper_freq - lower_freq
#             fwhm_arr.append(fwhm)
#             x_peaks.append(center_freq)
#             valid_mask[i] = True  # Mark this index as valid
            
#         elif np.all(lower_mask == False):
#             debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the lower edge")
#             continue 
            
#         elif np.all(upper_mask == False):
#             debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the upper edge")
#             continue 
        
#     return np.array(fwhm_arr), np.array(x_peaks), valid_mask

# post dave convo -- this will likely identify edges but thats ok
def full_width_half_max(x, y, peaks, half_maxes, verbose=False):
    fwhm_arr = np.empty_like(peaks)
    
    peakx = x[peaks]
    
    for i in range(len(peaks)):
        
        debug_print(verbose, "center freq", peakx[i])
        
        fwhm, low_x, high_x = get_a_fwhm(x, y, peakx[i], half_maxes[i], peaks[i])
        
        fwhm_arr[i] = fwhm
            
    return fwhm_arr

def wave_to_fwhms(wave, flux, sigma, poly, 
                  residual, coeff, 
                  max_diff=0.01, 
                  threshold_type="mad", interp_samples=None,
                  verbose=False):

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
    fwhms = full_width_half_max(wave, flux, peaks, half_maxes, verbose=verbose) # fwhm of peaks

    # debug_print(verbose, f"fwhm peaks: {len(x_peaks)}")
    
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
                      sigma=fwhm_G*broaden_coeff, 
                      gamma=fwhm_L*wl*broaden_coeff, 
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

# Claude optimized
def make_laser_arr(dir_path, 
                   wls, 
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
    
    # OPTIMIZATION 1: Load all combined_sigma files ONCE per observation
    combined_sigma_dict = {}
    for obsidx in range(n_obs):
        combined_sigma_dict[obsidx] = np.load(
            dir_path + f"/combined_sigma/combined_sigma_{obsidx}.npy"
        )
    
    # OPTIMIZATION 2: Pre-compute wavelength-to-pixel grids for set_fwhm
    wl_to_px_grids = {}
    if set_fwhm_px is not None:
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
    
    # OPTIMIZATION 3: Reorganize loop for better cache locality
    # Identify which orders contain each wavelength (vectorized)
    wl_to_orders = {}
    mins = np.nanmin(new_wave_arr, axis=(1, 2))
    maxs = np.nanmax(new_wave_arr, axis=(1, 2))
    
    for w, wl in enumerate(wls):
        orders = np.where((mins <= wl) & (maxs >= wl))[0]
        wl_to_orders[wl] = orders
        debug_print(verbose, f"WL {wl}: orders {orders}")
    
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
                    wave_of_px = wl_to_px_grids.get((order, obsidx))
                    if wave_of_px is not None:
                        g = np.poly1d(wave_of_px)
                        half_width = set_fwhm_px / 2
                        set_fwhm = g(col_idx + half_width) - g(col_idx - half_width)
                    else:
                        set_fwhm = None
                else:
                    set_fwhm = None
                
                # Evaluate LSF
                lsf_vals = lsf_per_wav(wave, 
                                       wl,
                                       amplitude_L=amplitude,
                                       broaden_coeff=broaden_coeff,
                                       set_fwhm=set_fwhm,
                                       model_type=model_type)
                
                laser_arr[order, :, obsidx] += lsf_vals
    
    return laser_arr
    

def fwhm_test(wave, x_peaks, method="pixel", px_min=None, amplitude_L=1, broaden_coeff=1, model_type="astropy"):
    
    if method == "pixel":
        # pixel - wavelength function to convert fwhm to pixels
        pixels = np.arange(0, len(wave), 1)
        fit_coeffs = np.polyfit(pixels, wave, 1)
        wave_of_px = np.poly1d(fit_coeffs)
        
        fwhm_min = wave_of_px(px_min) - wave_of_px(0)

        return fwhm_min

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

# def extract_peaks_between_minima(wave, flux, sigma, center_wavelengths):
#     """
#     Extract peak regions bounded by local minima on both sides. 
#     THIS SHOULD BE SLOPE = 0 BUT OK FOR NOW
    
#     Args:
#         wave: 1D wavelength array
#         flux: 1D flux array
#         center_wavelengths: 1D array or list of central wavelength peak locations (actual wavelength values)
    
#     Returns:
#         peaks_list: list of tuples (wave_segment, flux_segment) for valid peaks
#         skipped_peaks: 1D array of central wavelengths where minima weren't found on both sides
#     """
#     peaks_list = []
#     skipped_peaks = []
    
#     # Find all local minima in the spectrum (inverted flux)
#     minima_indices, _ = find_peaks(-flux)
#     minima_waves = wave[minima_indices]
    
#     for center_wave in center_wavelengths:
#         # Find minima on the left and right of this peak
#         left_minima = minima_waves[minima_waves < center_wave]
#         right_minima = minima_waves[minima_waves > center_wave]
        
#         # Check if both sides have a minimum
#         if len(left_minima) == 0 or len(right_minima) == 0:
#             skipped_peaks.append(center_wave)
#             continue
        
#         # Get the closest minimum on each side
#         left_min_wave = left_minima[-1]  # rightmost of left minima
#         right_min_wave = right_minima[0]  # leftmost of right minima
        
#         # Extract the segment (inclusive)
#         mask = (wave >= left_min_wave) & (wave <= right_min_wave)
#         wave_segment = wave[mask]
#         flux_segment = flux[mask]
#         sigma_segment = sigma[mask]
        
#         peaks_list.append((wave_segment, flux_segment, sigma_segment))
    
#     return peaks_list, np.array(skipped_peaks)

def thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose):
    
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
                                           verbose=verbose)
    
    lsf_fwhms = fwhm_test(wave, x_peaks, method=method, px_min=px_min)
    fwhm_test_pass = fwhms[fwhms > lsf_fwhms] 
    # doesnt work with method"model"
    x_test_pass = x_peaks[fwhms > lsf_fwhms]

    return (fwhms, x_peaks, 
            half_maxes, flx_pks, 
            threshold, 
            wave, flux, 
            poly, residual, 
            lsf_fwhms, 
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
            lsf_fwhms, 
            fwhm_test_pass, 
            x_test_pass) = thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose)
            
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
                'lsf_fwhms': lsf_fwhms,
                'fwhm_test_pass': fwhm_test_pass,
                'x_test_pass': x_test_pass
            }

    if save_dir:
        np.savez(savedir + "/peaks_arr.npz", fwhm_arr)

    return fwhm_arr

# this is recovery rate as a function of the allowed gap between expected and recovered
# def recovery_rate(fwhm_arr, tolerances=None,
#                   wls=None, x_peaks_key='x_peaks'):

#     if wls is None:
#         wls = np.arange(5200, 10400, 50)
#     if tolerances is None:
#         tolerances = np.logspace(-4, 0)
        
#     all_x_peaks_arr = np.concatenate([fwhm_arr[order, ordidx][x_peaks_key] 
#                             for order in range(fwhm_arr.shape[0]) 
#                             for ordidx in range(fwhm_arr.shape[1])])
    
#     recovereds = []
#     for tolerance in tolerances:
#         matched_fwhms = []
#         for target_wl in wls:
#             # Find FWHMs within tolerance
#             mask = np.abs(all_x_peaks_arr - target_wl) <= tolerance
#             if np.any(mask):
#                 # Pick closest among valid candidates
#                 valid_indices = np.where(mask)[0]
#                 closest_idx = valid_indices[np.argmin(np.abs(all_x_peaks_arr[valid_indices] - target_wl))]
#                 matched_fwhms.append(all_x_peaks_arr[closest_idx])
#             else:
#                 matched_fwhms.append(np.nan)  # No match within tolerance
        
#         matched_fwhms = np.array(matched_fwhms)
#         n_recovered = len(matched_fwhms[~np.isnan(matched_fwhms)])
#         recovereds.append(n_recovered)
    
#     recovereds = np.array(recovereds)
#     recovered_percentage = 100*recovereds / len(wls)
    
#     return tolerances, recovered_percentage


# def sample_recovery_rate(new_wave_arr, 
#                          normalized_spec, 
#                          poly_arr_best, 
#                          residual_arr, 
#                          normalized_sig, 
#                          coeff,
#                          n_runs=100, 
#                          wls=None, **kwargs):

#     mult = kwargs.get('mult', 1.5)
#     broaden_coeff = kwargs.get('broaden_coeff', 1)
#     set_fwhm_px = kwargs.get('set_fwhm_px', 2.5)
#     model_type = kwargs.get('model_type', 'astropy')

#     max_diff = kwargs.get('max_diff', 0.01)
#     threshold_type = kwargs.get('threshold_type', 'mad')
#     interp_samples = kwargs.get('interp_samples', 50000)

#     recovered_percentage_arr, recovered_percentage_pass_arr = [], []

#     fwhm_arrs = []
#     for run in tqdm(range(n_runs)):
#         if wls is None:
#             wls = np.random.uniform(low=5200, high=10400, size=(50,))
    
#         laser_arr = make_laser_arr(new_wave_arr, normalized_spec, poly_arr_best, 
#                                mult=mult, broaden_coeff=broaden_coeff, 
#                                    set_fwhm_px=set_fwhm_px, wls=wls,
#                                model_type=model_type)
        
#         normalized_laser_arr = laser_arr / poly_arr_best
        
#         fwhm_arr = get_fwhm_arr(new_wave_arr, normalized_laser_arr, 
#                          poly_arr_best, residual_arr, 
#                          normalized_sig, coeff, max_diff=max_diff, 
#                         threshold_type=threshold_type,
#                         interp_samples=interp_samples)
    
#         tolerances, recovered_percentage = recovery_rate(fwhm_arr, 
#                                                          wls=wls, 
#                                                          x_peaks_key='x_peaks')
#         (tolerances, 
#          recovered_percentage_pass) = recovery_rate(fwhm_arr, 
#                                                wls=wls,
#                                               x_peaks_key="x_test_pass")
    
#         recovered_percentage_arr.append(recovered_percentage)
#         recovered_percentage_pass_arr.append(recovered_percentage_pass)
#         fwhm_arrs.append(fwhm_arr)
        
#     recovered_percentage_arr = np.array(recovered_percentage_arr)
#     recovered_percentage_pass_arr = np.array(recovered_percentage_pass_arr)

#     return (tolerances, 
#             recovered_percentage_arr, 
#             recovered_percentage_pass_arr,
#             fwhm_arrs
#            )

def save_fwhm_per_obs(dir_path, save_folder,
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
                      all_data=False):
        
    norders = wave_arr.shape[0]
    nobs = wave_arr.shape[2]
 
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
            lsf_fwhms, 
            fwhm_test_pass, 
            x_test_pass) = thresh_and_fwhm(wave, flux, 
                    sigma, poly, 
                    residual, coeff, 
                    max_diff, 
                    threshold_type, 
                    interp_samples, 
                    method, px_min, 
                    verbose)
            
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
                    'lsf_fwhms': lsf_fwhms,
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

        save_path = dir_path + save_folder # e.g. "/base_peaks"
        if os.path.exists(save_path) is False: 
            os.mkdir(save_path)

        save_file = save_path + f"{save_folder}_{obsidx}.npz"
        np.savez(save_file, obs_arr)

        debug_print(verbose, ("saved to:", save_file))

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