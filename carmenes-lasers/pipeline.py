import astropy.units as u
import numpy as np

from astropy.modeling.models import Voigt1D
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
from tqdm import tqdm

from figures import debug_print

def check_spec_std(spec_arr, coeff=1):
    # Calculate std across wave_cols (axis 1) for all orders and observations
    std_array = np.nanstd(spec_arr, axis=1)  # shape: (orders, observations)
    
    std_mean = np.nanmean(std_array)
    std_std = np.std(std_array)
    
    # Return boolean array where True indicates std > threshold
    high_std_mask = std_array > (std_mean + std_std)
    return high_std_mask

def replace_nan_with_median(new_spec_arr):

    no_nan_spec_arr = new_spec_arr.copy()  # Don't modify original
    
    # Calculate median per (order, obs) pair, shape (orders, 1, observations)
    medians = np.nanmedian(new_spec_arr, axis=1, keepdims=True)
    
    # Replace non-finite values with median (broadcasts automatically)
    no_nan_spec_arr = np.where(np.isfinite(new_spec_arr), new_spec_arr, medians)
    
    return no_nan_spec_arr


# ___DOPPLER CORRECT FOR EARTH ORBIT___
def doppler_shift(wave_arr, velocity):
    """
    Doppler shift the wavelength array by a velocity (in m/s).
    
    Parameters
    ----------
    wave_arr: array
        1d wavelengths array (in any units)
    velocity: float
        Velocity in m/s

    Returns
    -------
    array 
        Doppler-shifted wavelengths
    """
    C = 299792458.0  # Speed of light in m/s
    return wave_arr * (1 + velocity / C)

def ds_wave_cube(wave_arr, bary_corr_arr):
    """
    Parameters
    ----------
    wave_arr: array of shape (order/detector, wave cols, observations)
    bary_corr_arr: 1d array of HELCOR values in km/s
    """
    shifted_wave_arr = np.empty_like(wave_arr)
    
    for order in range(len(wave_arr)):
        for obs in range(wave_arr.shape[2]):
            shifted_wave_arr[order, :, obs] = doppler_shift(wave_arr[order, :, obs], bary_corr_arr[obs] * 1000)

    return shifted_wave_arr

# ___RESAMPLE TO CONSISTENT WAVE GRID___
def resample(wave_arr, spec_arr, sig_arr, ordidx, new_wave=None, step=0.5,
             resampler="numpy", u_wav=u.angstrom, 
             u_flx=u.count):
    
    n_obs = spec_arr.shape[2]

    if new_wave is None:
        first_waves = np.array([wave_arr[ordidx, 0, i] for i in range(n_obs)])
        last_waves  = np.array([wave_arr[ordidx, -1, i] for i in range(n_obs)])
        
        max_first = step * np.ceil(first_waves.max() / step)
        min_last  = step * np.floor(last_waves.min() / step)
        wave = np.linspace(max_first, min_last, wave_arr.shape[1])

        new_wave = np.tile(wave[:, np.newaxis], (1, n_obs))
        
    new_spec = np.empty((len(new_wave), n_obs), dtype=float)
    new_sig = np.empty((len(new_wave), n_obs), dtype=float)

    if resampler == "fcr":
        fluxc_resample = FluxConservingResampler()
        wave_arr *= u_wav
        spec_arr *= u_flx
        sig_arr *= u_flx
        
        new_wave *= u_wav
        max_first *= u_wav
        min_last *= u_wav
    
    for i in range(n_obs):
        w = wave_arr[ordidx, :, i]
        s = spec_arr[ordidx, :, i]
        sig = sig_arr[ordidx, :, i]
    
        # restrict to the overlap region to avoid extrapolation
        m = (w >= max_first) & (w <= min_last)
    
        if resampler == "numpy":
            new_spec[:, i] = np.interp(new_wave[:, i], w[m], s[m])
            new_sig[:, i] = np.interp(new_wave[:, i], w[m], sig[m])
            
        if resampler == "fcr":
            input_spectra = Spectrum(flux=s[m], 
                         spectral_axis=w[m])
            input_sigma = Spectrum(flux=sig[m],
                                   spectral_axis=w[m])
            flux_resampled = fluxc_resample(input_spectra, 
                                            new_wave[:, i])
            sig_resampled = fluxc_resample(input_sigma,
                                          new_wave[:, i])
            new_spec[:, i] = flux_resampled.data
            new_sig[:, i] = sig_resampled.data
            
    return new_wave, new_spec, new_sig

def resample_ords(shifted_wave_arr, spec_arr, sig_arr, resampler="fcr", save_dir=None):
    new_wave_arr = np.empty_like(shifted_wave_arr)
    new_spec_arr = np.empty_like(shifted_wave_arr)
    new_sig_arr = np.empty_like(shifted_wave_arr)
    
    for i in tqdm(range(len(new_wave_arr))):
        new_wave, new_spec, new_sig = resample(shifted_wave_arr, spec_arr, sig_arr, i, resampler=resampler)
        new_wave_arr[i, :, :] = new_wave
        new_spec_arr[i, :, :] = new_spec
        new_sig_arr[i, :, :] = new_sig

    if save_dir:
        np.save(save_dir + "/resampled_wave.npy", new_wave_arr)
        np.save(save_dir + "/resampled_spec.npy", new_spec_arr)
        np.save(save_dir + "/resampled_sig.npy", new_sig_arr)

    return new_wave_arr, new_spec_arr, new_sig_arr

# ___REMOVE CONTINUUM__

def midpoints(arr):
    return (arr[:-1] + arr[1:]) / 2
    
def polyfit(wave, spectra, degree=4):

    z = np.polyfit(wave, spectra, degree)
    p = np.poly1d(z)

    return p(wave)

def bin_order(new_wave, no_nan_spec, bins=50, degree=4):
    bin_meds, bin_edges, binnumber = binned_statistic(new_wave, no_nan_spec, statistic='median', bins=bins)
    bin_midpts = midpoints(bin_edges)
    
    bin_poly = polyfit(bin_midpts, bin_meds, degree=degree)
    
    poly = np.interp(new_wave, bin_midpts, bin_poly)

    return poly, bin_midpts, bin_meds

def remove_polyfit(new_wave, new_spec, bins=50, degree=4):
    """
    only for 1 order!
    Parameters
    ----------
    new_spec: array of shape (wave cols, obs)
    """
    n_obs = new_spec.shape[1]

    poly = np.empty_like(new_spec)
    nopoly = np.empty_like(new_spec)

    bin_midpts = np.empty((bins, n_obs))
    bin_meds = np.empty((bins, n_obs))
    
    for obs in range(n_obs):
        if bins > 0: 
            poly[:, obs], bin_midpts[:, obs], bin_meds[:, obs] = bin_order(new_wave, 
                                       new_spec[:, obs], 
                                         bins=bins, 
                                       degree=degree)
        else: 
            poly[:, obs] = polyfit(new_wave, 
                                   new_spec[:, obs], 
                                   degree=degree)
        nopoly[:, obs] = new_spec[:, obs]/poly[:, obs]

    return poly, nopoly, bin_midpts, bin_meds

def remove_polyfit_orders(wave_arr, no_nan_spec_arr, bins=10, degree=4):
    n_ords = no_nan_spec_arr.shape[0]
    n_obs = no_nan_spec_arr.shape[2]
    
    poly_arr = np.empty_like(no_nan_spec_arr)
    nopoly_arr = np.empty_like(no_nan_spec_arr)

    bin_midpts_arr = np.empty((n_ords, bins, n_obs))
    bin_meds_arr = np.empty((n_ords, bins, n_obs))
    
    for i in range(n_ords):
        # print(f"Processing order {i}, shape: {no_nan_spec_arr[i].shape}")
        new_wave = wave_arr[i, :, 0]
        (poly_arr[i], 
         nopoly_arr[i], 
         bin_midpts_arr[i], 
         bin_meds_arr[i]) = remove_polyfit(new_wave, 
                                           no_nan_spec_arr[i], 
                                           bins=bins, 
                                           degree=degree)

    return poly_arr, nopoly_arr, bin_midpts_arr, bin_meds_arr

def chi2(x, mu, sig):
    unsummed = ((x - mu)/sig)**2
    return np.sum(unsummed, axis=1)

def bic_chi2(k, n, x, mu, sig):
    dof = n - k
    reduced_chi2 = chi2(x, mu, sig) / dof
    return k*np.log(n) + reduced_chi2
    
def find_best_poly(wave_arr, spec_arr, sig_arr, 
                   degrees=[1, 2, 3, 4, 5], 
                   bins=25, base_bic=1e6, 
                   verbose=False):

    n_ords = spec_arr.shape[0]
    n_obs = spec_arr.shape[2] 
    
    bic_vals = np.ones((n_ords, n_obs)) * base_bic
    deg_vals = np.zeros((n_ords, n_obs))  

    poly_arr_best = np.empty_like(spec_arr) 
    nopoly_arr_best = np.empty_like(spec_arr)
    bin_midpts_arr_best = np.empty((n_ords, bins, n_obs))
    bin_meds_arr_best = np.empty((n_ords, bins, n_obs))

    for degree in degrees:
        # Create mask: skip indices where best degree is 2+ below current
        # This means BIC already got worse, won't improve at higher degrees
        converged_mask = (deg_vals < degree - 1)

        mask_ord, mask_obs = np.where(converged_mask)
        spec_arr_masked = spec_arr.copy()
        sig_arr_masked = sig_arr.copy()
        
        # Use advanced indexing with full range for wavelengths
        spec_arr_masked[mask_ord[:, np.newaxis], :, mask_obs[:, np.newaxis]] = 0
        sig_arr_masked[mask_ord[:, np.newaxis], :, mask_obs[:, np.newaxis]] = np.inf

        (poly_arr, 
         nopoly_arr, 
         bin_midpts_arr, 
         bin_meds_arr) = remove_polyfit_orders(wave_arr, 
                                               spec_arr_masked, 
                                               bins=bins, degree=degree)

        k = degree + 1
        n = len(spec_arr[0, :, 0])
                    
        bic_chi = bic_chi2(k, n, spec_arr_masked, poly_arr, sig_arr_masked)
        
        debug_print(verbose, f"degree {degree} bic_chi:", bic_chi)

        # Element-wise comparison: where BIC improved
        improved = bic_chi < bic_vals

        # Update only improved indices
        bic_vals[improved] = bic_chi[improved]
        deg_vals[improved] = degree

        gord, gobs = np.where(improved)

        poly_arr_best[gord, :, gobs] = poly_arr[gord, :, gobs]
        nopoly_arr_best[gord, :, gobs] = nopoly_arr[gord, :, gobs]
        bin_midpts_arr_best[gord, :, gobs] = bin_midpts_arr[gord, :, gobs]
        bin_meds_arr_best[gord, :, gobs] = bin_meds_arr[gord, :, gobs]
        
        # Mark indices where BIC didn't improve as converged
        # (only if they weren't already marked converged)
        newly_converged = (~improved) & (~converged_mask)
        deg_vals[newly_converged] = degree - 1
        
        # If all indices have converged, exit early
        if np.all(improved == False):
            debug_print(verbose, f"All indices converged by degree {degree}")
            break

    return poly_arr_best, nopoly_arr_best, bin_midpts_arr_best, bin_meds_arr, bic_vals, deg_vals


# ___LASER DETECTION___
def simple_threshold(flux, coeff):
    return np.nanmedian(flux) + coeff * np.nanstd(flux)

def full_width_half_max(x, y, peakx, peaky, max_diff, verbose=False):
    fwhm_arr = []
    for i, peak in enumerate(peaky):
        half_max = peak/2
        center_freq = peakx[i]

        debug_print(verbose, "center freq", center_freq)
        x_args = np.where(np.abs(y - half_max) < max_diff)
        x_vals = x[x_args]
        # print("x vals", x_vals)
        debug_print(verbose, f"There are {len(x_vals)} intersections between the spectrum and half max of this peak.")
        lower_freq_arg = np.argmin(np.abs(center_freq - x_vals[x_vals < center_freq]))
        upper_freq_arg = np.argmin(np.abs(center_freq - x_vals[x_vals > center_freq]))

        lower_freq = x_vals[x_vals < center_freq][lower_freq_arg]
        upper_freq = x_vals[x_vals > center_freq][upper_freq_arg]
        debug_print(verbose, "lower, upper freqs", lower_freq, upper_freq)
        fwhm = upper_freq - lower_freq
        fwhm_arr.append(fwhm)
        
    return np.array(fwhm_arr)

def spec_to_fwhms(spec, flux, sigma, max_diff=0.01, verbose=False):
    threshold = simple_threshold(flux, sigma) # get the std threshold

    peaks, _ = find_peaks(flux, threshold) # get peaks above threshold 
    spec_pks, flx_pks = spec[peaks], flux[peaks]
    
    fwhms = full_width_half_max(spec, flux, spec_pks, flx_pks, max_diff, verbose=verbose) # fwhm of peaks

    return fwhms

def lsf_per_wav(spec, wl,
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

    return v1(spec)