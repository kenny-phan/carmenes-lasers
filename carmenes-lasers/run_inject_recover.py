import glob
import os

import numpy as np

from multiprocessing import Pool
from tqdm import tqdm

from laser import *

def get_inject_recover(dir_path, 
                       min_alpha=5, 
                       max_alpha=201, 
                       coeff=6, 
                       set_fwhm_px=2.5,
                       max_diff=0.25):
    
    star_name = dir_path.split("extracted/")[-1]
    
    results = np.load(dir_path + "/results.npz")

    wave_arr = results['new_wave_arr']
    flux_arr = results['normalized_spec']
    sigma_arr = results['normalized_sig']
    poly_arr = results['poly_arr_best']
    
    residual_arr = get_residual(flux_arr)
    
    alphas = np.arange(min_alpha, max_alpha, 1)
    
    for alpha in tqdm(alphas):
        print(f"Processing Star {star_name}, Alpha {alpha}")
        
        save_folder = f"/injection_alpha{alpha}"
    
        wls = generate_inj_params(length=250)
        
        laser_arr = make_laser_arr(dir_path, 
                                   wls,
                                   mult=alpha, set_fwhm_px=set_fwhm_px,
                                   model_type="astropy") 
        
        normalized_laser_arr = np.abs(laser_arr / poly_arr)
        injected_laser = flux_arr + normalized_laser_arr 

        inject_path = dir_path + "/injections"
        if os.path.exists(inject_path) is False: 
            os.mkdir(inject_path)
            
        save_fwhm_per_obs(inject_path, save_folder,
                              wave_arr, normalized_laser_arr, 
                              sigma_arr, poly_arr, 
                              residual_arr, mult=alpha, 
                              wls=wls, coeff=coeff, 
                              max_diff=max_diff, 
                              threshold_type='mad', 
                              interp_samples=50000, 
                              method='pixel', 
                              px_min=2.5, 
                              verbose=False)

# INPUT HERE
data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
dir_list = glob.glob(data_root + "extracted/*")
n_cpu = 20

if __name__ == "__main__":  # multiprocessing

    with Pool(n_cpu) as pool:  # create pool

        results = list(tqdm(
            pool.imap(get_inject_recover, [d for d in dir_list]),
            total=len(dir_list),
            desc="Processing stars"
        ))