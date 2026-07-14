import glob
import os

import numpy as np

from multiprocessing import Pool
from tqdm import tqdm

from laser import get_residual, thresh_and_fwhm

def process_peaks(dir_path, coeff=1, 
                  max_diff=0.01, 
                  threshold_type='mad', 
                  interp_samples=50000, 
                  method='pixel', 
                  px_min=2.5, 
                  verbose=False):

    star_name = dir_path.split("extracted/")[-1]

    results = np.load(dir_path + "/results.npz")

    wave_arr = results['new_wave_arr']
    flux_arr = results['normalized_spec']
    sigma_arr = results['normalized_sig']
    poly_arr = results['poly_arr_best']
    
    residual_arr = get_residual(flux_arr)

    norders = wave_arr.shape[0]
    nobs = wave_arr.shape[2]
 
    for obsidx in range(nobs): # per observation
        obs_arr = np.empty((norders), dtype=object)
        print(f"Processing Star {star_name}, Obs {obsidx}")

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
                'x_test_pass': x_test_pass
            }

        save_path = dir_path + "/base_peaks"
        if os.path.exists(save_path) is False: 
            os.mkdir(save_path)

        save_file = save_path + f"/base_peaks_{obsidx}.npz"
        np.savez(save_file, obs_arr)

        print("saved to:", save_file)

# INPUT HERE
data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
dir_list = glob.glob(data_root + "extracted/*")
n_cpu = 20

if __name__ == "__main__":  # multiprocessing

    with Pool(n_cpu) as pool:  # create pool

        results = list(tqdm(
            pool.imap(process_peaks, [d for d in dir_list]),
            total=len(dir_list),
            desc="Processing stars"
        ))

    
