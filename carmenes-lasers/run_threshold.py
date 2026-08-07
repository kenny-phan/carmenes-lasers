import glob
import os

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from laser import recovery_1d, get_ir_threshold

def save_ir_threshold(dir_path):
    results = np.load(dir_path + "/results.npz")
    
    results = np.load(dir_path + "/results.npz")
    wave_arr = results['new_wave_arr']
    
    all_alph = np.load(dir_path + "/injection.npz", allow_pickle=True)['arr_0']
    
    (plt_wls_recovered, plt_wls_not_recovered, 
     plt_mult_recovered, plt_mult_not_recovered) = recovery_1d(all_alph, wave_arr)

    min_alphas, filtered_wls = get_ir_threshold(plt_wls_recovered, plt_wls_not_recovered, 
                     plt_mult_recovered, plt_mult_not_recovered)

    np.savez(dir_path + "threshold.npz", alphas=min_alphas, wls=filtered_wls)


# INPUT HERE
data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
dir_list = glob.glob(data_root + "extracted/*")
n_cpu = 20

if __name__ == "__main__":  # multiprocessing

    with Pool(n_cpu) as pool:  # create pool

        results = list(tqdm(
            pool.imap(save_ir_threshold, [d for d in dir_list]),
            total=len(dir_list),
            desc="Processing stars"
        ))