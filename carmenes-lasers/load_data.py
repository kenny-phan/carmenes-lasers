import glob
import numpy as np
from astropy.io import fits

def debug_print(verbose, *args):
    if verbose:
        print(*args)

def check_channel_sizes(spec, cont, sigma, wave):
    if (spec.shape == cont.shape):
        if (spec.shape == sigma.shape):
            if (spec.shape == wave.shape):
                pass
            else:
                print("Warning! spec has a different shape from wave")
        else:
            print("Warning! spec has a different shape from sigma")
    else:
        print("Warning! spec has a different shape from wave")

def load_fits(filname, print_header=False, print_flux_header=False):
    with fits.open(filname) as hdul:
        spec = hdul[1].data
        cont = hdul[2].data
        sigma = hdul[3].data
        wave = hdul[4].data

        obj = hdul[0].header['OBJECT']
        ra = hdul[0].header['RA']
        dec = hdul[0].header['DEC']
        date = hdul[0].header['DATE-OBS']
        airm = hdul[0].header['AIRMASS']
        exptime = hdul[0].header['EXPTIME']
        
        try:
            bary_corr = hdul[0].header['HIERARCH CARACAL HELCOR']  # km/s
        except KeyError:
            try:
                bary_corr = hdul[0].header['HIERARCH CARACAL BERV']  # km/s (fallback)
            except KeyError as e:
                print(f"ERROR: No barycentric correction found in {filname}")
                bary_corr = 0.0

    check_channel_sizes(spec, cont, sigma, wave)

    if print_header:
        print(hdul[0].header)

    if print_flux_header:
        print(hdul[1].header)

    return spec, cont, sigma, wave, obj, ra, dec, date, exptime, airm, bary_corr

def load_star(dirname, print_headers=False, print_flux_headers=False):
    sci_list = glob.glob(dirname + "/*sci*.fits")

    spec_stack, cont_stack, sigma_stack, wave_stack, date_stack, exptime_stack, airm_stack = [], [], [], [], [], [], []
    obj_stack, ra_stack, dec_stack = [], [], []
    bary_corr_stack = []
    
    for sci in sci_list:
        spec, cont, sigma, wave, obj, ra, dec, date, exptime, airm, bary_corr = load_fits(sci, print_headers, print_flux_headers)
        spec_stack.append(spec)
        cont_stack.append(cont)
        sigma_stack.append(sigma)
        wave_stack.append(wave)
        date_stack.append(date)
        exptime_stack.append(exptime)
        airm_stack.append(airm)
        bary_corr_stack.append(bary_corr)

        ra_stack.append(ra)
        dec_stack.append(dec)
        obj_stack.append(obj)

    date_arr = np.array(date_stack, dtype="datetime64")
    sort_idx = np.argsort(date_arr)
    # keep date_arr as datetime64 sorted
    date_arr = date_arr[sort_idx]

    # reorder stacks using sort indices, then stack along axis=2
    spec_stack_sorted  = [spec_stack[i]  for i in sort_idx]
    cont_stack_sorted  = [cont_stack[i]  for i in sort_idx]
    sigma_stack_sorted = [sigma_stack[i] for i in sort_idx]
    wave_stack_sorted = [wave_stack[i] for i in sort_idx]
    exptime_stack_sorted = [exptime_stack[i] for i in sort_idx]
    airm_stack_sorted  = [airm_stack[i]  for i in sort_idx]
    ra_stack_sorted = [ra_stack[i] for i in sort_idx]
    dec_stack_sorted = [dec_stack[i] for i in sort_idx]
    obj_stack_sorted = [obj_stack[i] for i in sort_idx]
    bary_corr_stack_sorted = [bary_corr_stack[i] for i in sort_idx]

    spec_arr  = np.stack(spec_stack_sorted,  axis=2)
    cont_arr  = np.stack(cont_stack_sorted,  axis=2)
    sigma_arr = np.stack(sigma_stack_sorted, axis=2)
    wave_arr = np.stack(wave_stack_sorted, axis=2)
    exptime_arr = np.array(exptime_stack_sorted)
    airm_arr  = np.array(airm_stack_sorted)
    ra_arr = np.array(ra_stack_sorted)
    dec_arr = np.array(dec_stack_sorted)
    obj_arr = np.array(obj_stack_sorted)
    bary_corr_arr = np.array(bary_corr_stack_sorted)

    return spec_arr, cont_arr, sigma_arr, wave_arr, obj_arr, ra_arr, dec_arr, date_arr, exptime_arr, airm_arr, bary_corr_arr


    