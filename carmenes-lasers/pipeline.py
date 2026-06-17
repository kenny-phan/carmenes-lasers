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