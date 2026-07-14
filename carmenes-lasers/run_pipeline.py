# load in the carmenes data
import glob
import os

import numpy as np

from multiprocessing import Pool
from tqdm import tqdm

from load_data import load_star
from pipeline import resample_and_fit

def process_star(dir_path, dont_recompute=False, dont_resample=True):  # NEW: worker function

    print(f"Path: {dir_path}")
    
    star_name = dir_path.split("extracted/")[-1]
    
    if os.path.exists(dir_path + "/results.npz") and dont_recompute:
        print(f"Star {star_name} already processed. moving on.")
        return None
    
    print(f"Processing Star {star_name}")
    
    sci_list = glob.glob(dir_path + "/*sci*.fits")
    print(f"There are {len(sci_list)} observations")
    
    (spec_arr, 
     cont_arr, 
     sigma_arr, 
     wave_arr, 
     obj, 
     ra, 
     dec, 
     date_arr, 
     exptime_arr, 
     airm_arr, 
     bary_corr_arr) = load_star(dir_path, 
                                print_headers=False, 
                                print_flux_headers=False)

    print("Star loaded")
    
    results = resample_and_fit(wave_arr, 
                    spec_arr, 
                    sigma_arr,
                    bary_corr_arr, 
                    dir_path,
                    dont_resample=dont_resample,
                    degrees = np.arange(2, 6, 1),
                    criterion="AIC",
                    verbose=True)
    return results

# INPUT HERE
data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
stars_folder = "extracted/"
dir_list = glob.glob(data_root + stars_folder + "/*")
n_cpu = 20
dont_recompute = False 
dont_resample = True

if __name__ == "__main__":  # multiprocessing

    with Pool(n_cpu) as pool:  # create pool

        results = list(tqdm(
            pool.starmap(process_star, [(d, dont_recompute, dont_resample) for d in dir_list]),
            total=len(dir_list),
            desc="Processing stars"
        ))
