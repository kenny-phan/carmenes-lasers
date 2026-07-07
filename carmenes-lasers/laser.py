import numpy as np

from astropy.modeling.models import Voigt1D
from scipy import ndimage
from scipy.signal import find_peaks

from load_data import debug_print 

# ___LASER DETECTION___

def med_abs_dev(x):
    med = np.nanmedian(x)
    abs_dev = np.abs(x - med)
    mad = np.nanmedian(abs_dev)
    return mad

def get_residual(spec_arr):
    """Subtract median spectrum from each observation"""
    median_obs = np.nanmedian(spec_arr, axis=2)  # Shape: (ords, wave_cols)
    
    # Broadcasting: (ords, wave_cols, obs) - (ords, wave_cols, 1)
    residual_arr = spec_arr - median_obs[:, :, np.newaxis]
    
    return residual_arr
    
def full_width_half_max(x, y, peakx, peaky, max_diff, verbose=False):
    fwhm_arr = []
    x_peaks = []
    for i, peak in enumerate(peaky):
        half_max = peak/2
        center_freq = peakx[i]

        debug_print(verbose, "center freq", center_freq)

        x_args = np.where(np.abs(y - half_max) < max_diff)
        x_vals = x[x_args]

        debug_print(verbose, 
                    f"{len(x_vals)} intersections between the spectrum and half max of this peak.")

        lower_mask = (x_vals < center_freq)
        upper_mask = (x_vals > center_freq)

        if np.any(lower_mask == True) and np.any(upper_mask == True):
            lower_freq_arg = np.argmin(np.abs(center_freq - x_vals[lower_mask]))
            upper_freq_arg = np.argmin(np.abs(center_freq - x_vals[upper_mask]))
    
            lower_freq = x_vals[lower_mask][lower_freq_arg]
            upper_freq = x_vals[upper_mask][upper_freq_arg]
            debug_print(verbose, "lower, upper freqs", lower_freq, upper_freq)
            fwhm = upper_freq - lower_freq
            fwhm_arr.append(fwhm)
            x_peaks.append(center_freq)
            
        elif np.all(lower_mask == False):
            debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the lower edge")
            continue 
            
        elif np.all(upper_mask == False):
            debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the upper edge")
            continue 
        
    return np.array(fwhm_arr), np.array(x_peaks)

def spec_to_fwhms(wave, flux, poly, residual, coeff, max_diff=0.01, threshold_type="mad", verbose=False):
    
    if threshold_type == "std":
        threshold = poly + coeff * np.nanstd(residual) 
    elif threshold_type == "mad":
        threshold = poly + coeff * med_abs_dev(residual)

    peaks, _ = find_peaks(flux, threshold) # get peaks above threshold 
    wave_pks, flx_pks = wave[peaks], flux[peaks]
    
    fwhms, x_peaks = full_width_half_max(wave, flux, wave_pks, flx_pks, max_diff, verbose=verbose) # fwhm of peaks

    return fwhms, x_peaks, threshold

def lsf_per_wav(wave, wl,
                amplitude_L=1, 
                w_lorentz=np.array([0.28, 0.21, 0.17]) * 1e-5, 
                w_gauss=np.array([1.0, 1.01, 1.18]) * 1e-5):
    
    if wl <= 6000:
        windex = 0
    elif (wl > 6000) and (wl <= 9600):
        windex = 1
    elif wl > 9600:
        windex = 2

    v1 = Voigt1D(x_0=wl, 
                 amplitude_L=amplitude_L, 
                 fwhm_L=w_lorentz[windex]*wl, 
                 fwhm_G=w_gauss[windex]*wl)

    return v1(wave)

# def identify_peaks(normalized_spec, poly_arr_best, n=3):
#     """
#     Adapted from Tellis & Marcy 2017, Sec. 3 
#     """
#     num_orders = normalized_spec.shape[0]
#     num_obs = normalized_spec.shape[2]
    
#     sub_arr = normalized_spec - poly_arr_best
    
#     positive_sub_arr = np.copy(sub_arr)
#     positive_sub_arr[positive_sub_arr < 0] = np.nan
    
#     positive_sub_arr_filtered = np.copy(positive_sub_arr)
#     medians_dict = {}  # Store medians indexed by (order, obs, group_id)

#     median_of_medians = np.full((num_orders, num_obs), np.nan)
    
#     for order in range(num_orders):
#         for obs in range(num_obs):
#             spectrum = positive_sub_arr_filtered[order, :, obs]
#             labeled, num_f = ndimage.label(~np.isnan(spectrum))
            
#             if num_f > 0:
#                 sizes = np.bincount(labeled)[1:]  # Skip background (0)
#                 valid = np.where(sizes > n)[0] + 1
                
#                 # Calculate median for each valid group
#                 for group_id in valid:
#                     group_pixels = spectrum[labeled == group_id]
#                     group_median = np.nanmedian(group_pixels)
#                     medians_dict[(order, group_id, obs)] = group_median
                
#                 # Set invalid groups to NaN
#                 positive_sub_arr_filtered[order, :, obs][~np.isin(labeled, valid)] = np.nan
#                 medians_for_pair = [v for (o, g, ob), v in medians_dict.items() 
#                            if o == order and ob == obs]

#                 if medians_for_pair:
#                     median_of_medians[order, obs] = np.median(medians_for_pair)

#     return positive_sub_arr_filtered, median_of_medians

def make_laser_arr(new_wave_arr, 
                   normalized_spec,
                   poly_arr_best, 
                   wls=None, 
                   mult=1.5, 
                   n=3, 
                   verbose=False):
    if wls is None:
        wls = np.arange(5200, 10400, 50)
    
    n_ords, n_cols, n_obs = new_wave_arr.shape
    laser_arr = np.zeros((n_ords, n_cols, n_obs))
    
    for wl in wls:
        # Find which orders contain this wavelength
        mins = np.nanmin(new_wave_arr, axis=(1, 2))
        maxs = np.nanmax(new_wave_arr, axis=(1, 2))
        orders = np.where((mins <= wl) & (maxs >= wl))[0]
        
        debug_print(verbose, f"WL {wl}: orders {orders}")
        
        for order in orders:
            for obsidx in range(n_obs):
                # Find column index closest to target wavelength in this order/obs
                wl_cols = new_wave_arr[order, :, obsidx]
                col_idx = np.nanargmin(np.abs(wl_cols - wl))  # Closest column
                
                amplitude = (poly_arr_best[order, col_idx, obsidx]                             * mult)
                
                laser_arr[order, :, obsidx] += lsf_per_wav(wl_cols, 
                                                       wl,
                                                       amplitude_L=amplitude)
    
    return laser_arr
