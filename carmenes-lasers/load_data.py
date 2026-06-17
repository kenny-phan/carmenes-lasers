import glob
import numpy as np
from astropy.io import fits

from figures import debug_print

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

def load_fits(filname, print_header=False):
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

    check_channel_sizes(spec, cont, sigma, wave)

    if print_header:
        print(hdul[0].header)

    return spec, cont, sigma, wave, obj, ra, dec, date, exptime, airm

def load_star(dirname):
    sci_list = glob.glob(dirname + "/*sci*.fits")

    spec_stack, cont_stack, sigma_stack, date_stack, exptime_stack, airm_stack = [], [], [], [], [], []
    
    for sci in sci_list:
        spec, cont, sigma, wave, obj, ra, dec, date, exptime, airm = load_fits(sci)
        spec_stack.append(spec)
        cont_stack.append(cont)
        sigma_stack.append(sigma)
        date_stack.append(date)
        exptime_stack.append(exptime)
        airm_stack.append(airm)

    date_arr = np.array(date_stack, dtype="datetime64")
    sort_idx = np.argsort(date_arr)
    # keep date_arr as datetime64 sorted
    date_arr = date_arr[sort_idx]

    # reorder stacks using sort indices, then stack along axis=2
    spec_stack_sorted  = [spec_stack[i]  for i in sort_idx]
    cont_stack_sorted  = [cont_stack[i]  for i in sort_idx]
    sigma_stack_sorted = [sigma_stack[i] for i in sort_idx]
    exptime_stack_sorted = [exptime_stack[i] for i in sort_idx]
    airm_stack_sorted  = [airm_stack[i]  for i in sort_idx]

    spec_arr  = np.stack(spec_stack_sorted,  axis=2)
    cont_arr  = np.stack(cont_stack_sorted,  axis=2)
    sigma_arr = np.stack(sigma_stack_sorted, axis=2)
    exptime_arr = np.array(exptime_stack_sorted)
    airm_arr  = np.array(airm_stack_sorted)

    return spec_arr, cont_arr, sigma_arr, wave, obj, ra, dec, date_arr, exptime_arr, airm_arr


    