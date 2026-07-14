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
    
def med_abs_dev(x):
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
    
def full_width_half_max(x, y, peakx, half_maxes, max_diff=0.01, verbose=False):
    fwhm_arr = []
    x_peaks = []
    valid_mask = np.zeros(len(half_maxes), dtype=bool)  # True = peak passed, False = excluded
    
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
            valid_mask[i] = True  # Mark this index as valid
            
        elif np.all(lower_mask == False):
            debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the lower edge")
            continue 
            
        elif np.all(upper_mask == False):
            debug_print(verbose, f"peak at {np.round(center_freq,2)} is at the upper edge")
            continue 
        
    return np.array(fwhm_arr), np.array(x_peaks), valid_mask
    

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
        
    wave_pks, flx_pks, poly_pks = wave[peaks], flux[peaks], poly[peaks]

    half_maxes = poly_pks + 0.5*(flx_pks - poly_pks)
    
    fwhms, x_peaks, valid_mask = full_width_half_max(wave, flux, 
                                         wave_pks, half_maxes, 
                                         max_diff, verbose=verbose) # fwhm of peaks

    return (fwhms, x_peaks, half_maxes[valid_mask], 
            flx_pks[valid_mask], threshold, 
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
                   n=3, broaden_coeff=1,
                   model_type="astropy",
                   set_fwhm_px=None,
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
                wave = new_wave_arr[order, :, obsidx]
                col_idx = np.nanargmin(np.abs(wave - wl))  # Closest column
                
                amplitude = (poly_arr_best[order, col_idx, obsidx] 
                             * mult)

                if set_fwhm_px is not None:
                    pixels = np.linspace(0, len(wave), len(wave))
                    wave_of_px = np.polyfit(pixels, wave, 1)
                    g = np.poly1d(wave_of_px)

                    half_width = set_fwhm_px / 2

                    set_fwhm = g(col_idx + half_width) - g(col_idx - half_width)
                    
                laser_arr[order, :, obsidx] += lsf_per_wav(wave, 
                                                       wl,
                                                       amplitude_L=amplitude,
                                                       broaden_coeff=broaden_coeff,
                                                       set_fwhm=set_fwhm,
                                                       model_type=model_type)
    
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


# default args above
# 1-10s runtime
def get_fwhm_arr(new_wave_arr, flux_arr, 
                 poly_arr_best, residual_arr, 
                 normalized_sig, coeff, **kwargs):

    max_diff = kwargs.get('max_diff', 0.01)
    threshold_type = kwargs.get('threshold_type', 'std')
    interp_samples = kwargs.get('interp_samples', None)
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', "pixel")
    px_min = kwargs.get('px_min', 2.5)
    
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

    return fwhm_arr


def recovery_rate(fwhm_arr, tolerances=None,
                  wls=None, x_peaks_key='x_peaks'):

    if wls is None:
        wls = np.arange(5200, 10400, 50)
    if tolerances is None:
        tolerances = np.logspace(-4, 0)
        
    all_x_peaks_arr = np.concatenate([fwhm_arr[order, ordidx][x_peaks_key] 
                            for order in range(fwhm_arr.shape[0]) 
                            for ordidx in range(fwhm_arr.shape[1])])
    
    recovereds = []
    for tolerance in tolerances:
        matched_fwhms = []
        for target_wl in wls:
            # Find FWHMs within tolerance
            mask = np.abs(all_x_peaks_arr - target_wl) <= tolerance
            if np.any(mask):
                # Pick closest among valid candidates
                valid_indices = np.where(mask)[0]
                closest_idx = valid_indices[np.argmin(np.abs(all_x_peaks_arr[valid_indices] - target_wl))]
                matched_fwhms.append(all_x_peaks_arr[closest_idx])
            else:
                matched_fwhms.append(np.nan)  # No match within tolerance
        
        matched_fwhms = np.array(matched_fwhms)
        n_recovered = len(matched_fwhms[~np.isnan(matched_fwhms)])
        recovereds.append(n_recovered)
    
    recovereds = np.array(recovereds)
    recovered_percentage = 100*recovereds / len(wls)
    
    return tolerances, recovered_percentage


def sample_recovery_rate(new_wave_arr, 
                         normalized_spec, 
                         poly_arr_best, 
                         residual_arr, 
                         normalized_sig, 
                         coeff,
                         n_runs=100, 
                         wls=None, **kwargs):

    mult = kwargs.get('mult', 1.5)
    broaden_coeff = kwargs.get('broaden_coeff', 1)
    set_fwhm_px = kwargs.get('set_fwhm_px', 2.5)
    model_type = kwargs.get('model_type', 'astropy')

    max_diff = kwargs.get('max_diff', 0.01)
    threshold_type = kwargs.get('threshold_type', 'mad')
    interp_samples = kwargs.get('interp_samples', 50000)

    recovered_percentage_arr, recovered_percentage_pass_arr = [], []

    fwhm_arrs = []
    for run in tqdm(range(n_runs)):
        if wls is None:
            wls = np.random.uniform(low=5200, high=10400, size=(50,))
    
        laser_arr = make_laser_arr(new_wave_arr, normalized_spec, poly_arr_best, 
                               mult=mult, broaden_coeff=broaden_coeff, 
                                   set_fwhm_px=set_fwhm_px, wls=wls,
                               model_type=model_type)
        
        normalized_laser_arr = laser_arr / poly_arr_best
        
        fwhm_arr = get_fwhm_arr(new_wave_arr, normalized_laser_arr, 
                         poly_arr_best, residual_arr, 
                         normalized_sig, coeff, max_diff=max_diff, 
                        threshold_type=threshold_type,
                        interp_samples=interp_samples)
    
        tolerances, recovered_percentage = recovery_rate(fwhm_arr, 
                                                         wls=wls, 
                                                         x_peaks_key='x_peaks')
        (tolerances, 
         recovered_percentage_pass) = recovery_rate(fwhm_arr, 
                                               wls=wls,
                                              x_peaks_key="x_test_pass")
    
        recovered_percentage_arr.append(recovered_percentage)
        recovered_percentage_pass_arr.append(recovered_percentage_pass)
        fwhm_arrs.append(fwhm_arr)
        
    recovered_percentage_arr = np.array(recovered_percentage_arr)
    recovered_percentage_pass_arr = np.array(recovered_percentage_pass_arr)

    return (tolerances, 
            recovered_percentage_arr, 
            recovered_percentage_pass_arr,
            fwhm_arrs
           )