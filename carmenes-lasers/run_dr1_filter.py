import glob
import os

import numpy as np

from multiprocessing import Pool
from tqdm import tqdm

def process_pass(dir_path):
    
    uni_threshold = np.load('/datax/scratch/ktp/carmenes-lasers/universal_threshold.npz')
    uni_alpha = uni_threshold['alpha']
    uni_wl = uni_threshold['wave']

    peaks_path = dir_path + "/base_peaks"
    peaks_list = glob.glob(peaks_path + "/*")

    for obsidx, peak_file in enumerate(peaks_list):
    
        # make folder
        save_path = dir_path + "/peaks_pass"
        if os.path.exists(save_path) is False: 
            os.mkdir(save_path)
            
        peak_data = np.load(peak_file, allow_pickle=True)['arr_0']
        nords = len(peak_data)
        # print(f"{nords} orders")
        obs_flx_pass = np.empty((nords), dtype=object)
        obs_wave_pass = np.copy(obs_flx_pass)
        # print(len(obs_flx_pass))
        for ordidx, order_data in enumerate(peak_data):
    
            wave = order_data['wave']
            # print(f"wave shape: {wave.shape}")
            threshold = order_data['threshold']
            wave_peaks = order_data['x_test_pass']
            # print(f"wave peaks shape: {wave_peaks.shape}")

            fwhms = order_data['fwhms']
            # print(f"fwhm shape: {fwhms.shape}")
            flx_pks = order_data['flx_pks']
            min_lsf_fwhms = order_data['min_lsf_fwhms']
            max_lsf_fwhms = order_data['max_lsf_fwhms']
    
            fwhm_mask = (fwhms > min_lsf_fwhms) & (fwhms < max_lsf_fwhms)
            flux_peaks_masked = flx_pks[fwhm_mask]
    
            ord_threshold = np.interp(wave, uni_wl, uni_alpha) * threshold
            ord_threshold_masked = np.interp(wave_peaks, wave, ord_threshold)
            
            obs_flx_pass[ordidx] = flux_peaks_masked[flux_peaks_masked > ord_threshold_masked]
            obs_wave_pass[ordidx] = wave_peaks[flux_peaks_masked > ord_threshold_masked]
    
        np.savez(save_path + f"/pass_peaks_{obsidx}.npz", wave=obs_wave_pass, flux=obs_flx_pass)
    
# INPUT HERE
data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
dir_list = glob.glob(data_root + "extracted/*")
n_cpu = 20

if __name__ == "__main__":  # multiprocessing

    with Pool(n_cpu) as pool:  # create pool

        results = list(tqdm(
            pool.imap(process_pass, [d for d in dir_list]),
            total=len(dir_list),
            desc="Processing stars"
        ))
