import numpy as np

from astropy.modeling.models import Voigt1D
from lmfit.models import VoigtModel
# from scipy import ndimage
from scipy.signal import find_peaks

from load_data import debug_print 

# ___LASER DETECTION___
def mse(obs, pred):
    return np.sum((obs - pred)**2) / len(obs)
    
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
    
def full_width_half_max(x, y, peakx, half_maxes, max_diff=0.01, verbose=False):
    fwhm_arr = []
    x_peaks = []
    for i, half_max in enumerate(half_maxes):
        # half_max = peak/2
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
        
    wave_pks, flx_pks, poly_pks = wave[peaks], flux[peaks], poly[peaks]

    half_maxes = poly_pks + 0.5*(flx_pks - poly_pks)
    
    fwhms, x_peaks = full_width_half_max(wave, flux, wave_pks, half_maxes, max_diff, verbose=verbose) # fwhm of peaks

    return fwhms, x_peaks, threshold

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
               model_type="astropy"):
    
    windex = check_windex(wl)

    if model_type == "astropy":
        v1 = Voigt1D(x_0=wl, 
                     amplitude_L=amplitude_L, 
                     fwhm_L=w_lorentz[windex]*wl, 
                     fwhm_G=w_gauss[windex]*wl)

        return v1(wave)
        
    if model_type == "lmfit":
        model = VoigtModel()
        v1 = model.eval(amplitude=amplitude_L, 
                      center=wl, 
                      sigma=w_gauss[windex]*wl, 
                      gamma=w_lorentz[windex]*wl, 
                      x=wave)
        return v1

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
                   n=3, model_type="astropy",
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
                
                amplitude = (poly_arr_best[order, col_idx, obsidx] 
                             * mult)
                
                laser_arr[order, :, obsidx] += lsf_per_wav(wl_cols, 
                                                       wl,
                                                       amplitude_L=amplitude,
                                                       model_type=model_type)
    
    return laser_arr

def fwhm_test(wave, x_peaks, method="pixel", px_min=None):
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
                            amplitude_L=1)
            
                fwhm, _, _ = spec_to_fwhms(wave, lsfs[windex, :], 
                                               np.zeros_like(wave), 
                                               np.zeros_like(wave), 
                                               0, max_diff=0.5)
                fwhm_lsfs[windex] = fwhm[0]
                
                unfilled_ranges[windex] = False
        
            else:
                continue

        return fwhm_lsfs

def extract_peaks_between_minima(wave, flux, sigma, center_wavelengths):
    """
    Extract peak regions bounded by local minima on both sides. 
    THIS SHOULD BE SLOPE = 0 BUT OK FOR NOW
    
    Args:
        wave: 1D wavelength array
        flux: 1D flux array
        center_wavelengths: 1D array or list of central wavelength peak locations (actual wavelength values)
    
    Returns:
        peaks_list: list of tuples (wave_segment, flux_segment) for valid peaks
        skipped_peaks: 1D array of central wavelengths where minima weren't found on both sides
    """
    peaks_list = []
    skipped_peaks = []
    
    # Find all local minima in the spectrum (inverted flux)
    minima_indices, _ = find_peaks(-flux)
    minima_waves = wave[minima_indices]
    
    for center_wave in center_wavelengths:
        # Find minima on the left and right of this peak
        left_minima = minima_waves[minima_waves < center_wave]
        right_minima = minima_waves[minima_waves > center_wave]
        
        # Check if both sides have a minimum
        if len(left_minima) == 0 or len(right_minima) == 0:
            skipped_peaks.append(center_wave)
            continue
        
        # Get the closest minimum on each side
        left_min_wave = left_minima[-1]  # rightmost of left minima
        right_min_wave = right_minima[0]  # leftmost of right minima
        
        # Extract the segment (inclusive)
        mask = (wave >= left_min_wave) & (wave <= right_min_wave)
        wave_segment = wave[mask]
        flux_segment = flux[mask]
        sigma_segment = sigma[mask]
        
        peaks_list.append((wave_segment, flux_segment, sigma_segment))
    
    return peaks_list, np.array(skipped_peaks)