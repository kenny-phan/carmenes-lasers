import astropy.units as u
import numpy as np

from astropy.modeling.models import Voigt1D
from scipy.signal import find_peaks
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
from tqdm import tqdm

from figures import debug_print

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

# ___DOPPLER CORRECT FOR EARTH ORBIT___
def doppler_shift(wave_arr, velocity):
    """
    Doppler shift the wavelength array by a velocity (in m/s).
    
    Parameters
    ----------
    wave_arr: array
        1d wavelengths array (in any units, e.g., nm or µm)
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
def resample(wave_arr, spec_arr, ordidx, new_wave=None, step=0.5,
             resampler="numpy", u_wav=u.angstrom, 
             u_flx=u.count):
    
    n_obs = spec_arr.shape[2]

    if new_wave is None:
        first_waves = np.array([wave_arr[ordidx, 0, i] for i in range(n_obs)])
        last_waves  = np.array([wave_arr[ordidx, -1, i] for i in range(n_obs)])
        
        max_first = step * np.ceil(first_waves.max() / step)
        min_last  = step * np.floor(last_waves.min() / step)
        new_wave = np.linspace(max_first, min_last, wave_arr.shape[1])
    
    new_spec_arr = np.empty((len(new_wave), n_obs), dtype=float)
        
    if resampler == "fcr":
        fluxc_resample = FluxConservingResampler()
        wave_arr *= u_wav
        spec_arr *= u_flx
        new_wave *= u_wav
        max_first *= u_wav
        min_last *= u_wav
    
    for i in range(n_obs):
        w = wave_arr[ordidx, :, i]
        s = spec_arr[ordidx, :, i]
    
        # restrict to the overlap region to avoid extrapolation
        m = (w >= max_first) & (w <= min_last)
    
        if resampler == "numpy":
            new_spec_arr[:, i] = np.interp(new_wave, w[m], s[m])
            
        if resampler == "fcr":
            input_spectra = Spectrum(flux=s[m], 
                         spectral_axis=w[m])
            flux_resampled = fluxc_resample(input_spectra, 
                                            new_wave)
            new_spec_arr[:, i] = flux_resampled.data

    return new_wave, new_spec_arr

def resample_ords(shifted_wave_arr, spec_arr, resampler="fcr", save_dir=None):
    new_wave_arr = np.empty_like(shifted_wave_arr[:, :, 0])
    new_spec_arr = np.empty_like(shifted_wave_arr)
    
    for i in tqdm(range(len(new_wave_arr))):
        new_wave, new_spec = resample(shifted_wave_arr, spec_arr, i, resampler=resampler)
        new_wave_arr[i, :] = new_wave
        new_spec_arr[i, :, :] = new_spec

    if save_dir:
        np.save(save_dir + "wave_grid.npy", new_wave_arr)
        np.save(save_dir + "resampled_spec.npy", new_spec_arr)

    return new_wave_arr, new_spec_arr

# ___REMOVE CONTINUUM__
def polyfit(wave, spectra, degree=4):
    med = np.nanmedian(spectra)
    nonan_spectra = np.where(np.isfinite(spectra), spectra, med)  # replace NaN/Inf with median

    z = np.polyfit(wave, nonan_spectra, degree)
    p = np.poly1d(z)

    return p(wave)

def remove_polyfit(new_wave, new_spec_arr, degree=4):
    """
    only for 1 order!
    Parameters
    ----------
    new_spec_arr: array of shape (wave cols, obs)
    """
    poly_arr = np.empty_like(new_spec_arr)
    nopoly_arr = np.empty_like(new_spec_arr)
    for obs in range(new_spec_arr.shape[1]):
        poly_arr[:, obs] = polyfit(new_wave, 
                                   new_spec_arr[:, obs], 
                                   degree=degree)
        nopoly_arr[:, obs] = new_spec_arr[:, obs]/poly_arr[:, obs]

    return poly_arr, nopoly_arr