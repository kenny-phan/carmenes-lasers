# load in the carmenes data
import glob

from load_data import load_star
from pipeline import resample_and_fit

data_root = "/datax/scratch/ktp/carmenes-lasers/spectra/"
stars_folder = "extracted/"
dir_list = glob.glob(data_root + stars_folder + "/*")

for diridx in range(len(dir_list)):

    print(f"Processing Star {dir_list[diridx].split(stars_folder)[-1]}")

    sci_list = glob.glob(dir_list[diridx] + "/*sci*.fits")
    
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
     bary_corr_arr) = load_star(dir_list[diridx], 
                                print_headers=False, 
                                print_flux_headers=False)

    print("Star loaded")
    
    results = resample_and_fit(wave_arr, 
                    spec_arr, 
                    sigma_arr,
                    bary_corr_arr, 
                    dir_list[diridx],
                    criterion="AIC",
                    verbose=True)