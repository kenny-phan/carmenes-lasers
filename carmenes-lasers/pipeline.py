import numpy as np
from astropy.modeling.models import Voigt1D

from figures import debug_print

def simple_threshold(flux, coeff):
    return np.median(flux) + coeff * np.std(flux)

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

def lsf_per_wav(spec, spec_peaks,
                amplitude_L=1, 
                w_lorentz=np.array([0.28, 0.21, 0.17]) * 1e-5, 
                w_gauss=np.array([1.0, 1.01, 1.18]) * 1e-5):
    
    for wl in spec_peaks:
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

def numpy_resample(wave_arr, spec_arr, ordidx, step=0.5):
    n_obs = spec_arr.shape[2]
    n_cols = spec_arr.shape[1]
    
    # collect wave endpoints across all observations
    first_waves = np.empty(n_obs)
    last_waves  = np.empty(n_obs)
    
    for i in range(n_obs):
        w = wave_arr[ordidx, :, i]
        first_waves[i] = w[0]
        last_waves[i]  = w[-1]
    
    # rounding to nearest step:
    max_first = step * np.ceil(first_waves.max() / step)   # up for wave[0]
    min_last  = step * np.floor(last_waves.min() / step)   # down for wave[-1]
    
    new_wave = np.linspace(max_first, min_last, n_cols)
    
    # resample each spectrum onto new_wave
    new_spec_arr = np.empty((n_cols, n_obs), dtype=float)
    
    for i in range(n_obs):
        w = wave_arr[ordidx, :, i]
        s = spec_arr[ordidx, :, i]
    
        # restrict to the overlap region to avoid extrapolation
        m = (w >= max_first) & (w <= min_last)
    
        # if your wave grid is monotonic increasing, np.interp works directly
        new_spec_arr[:, i] = np.interp(new_wave, w[m], s[m])

    return new_wave, new_spec_arr