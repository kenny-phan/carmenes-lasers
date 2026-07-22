import astropy.units as u
import numpy as np

from astropy.constants import L_sun

def parallax_to_distance(parallax): # in arcsec
    distance = (1 / parallax.value)
    return distance * u.parsec

def plax_err_to_dist_err(parallax, parallax_err):
    dist_err = (parallax_err.value / parallax.value**2)
    return dist_err * u.parsec

# per: 
# https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html
def flux_to_app_mag(flux, zp=25.6884): # Gaia G-Band zero point
    mag = zp - 2.5*np.log10(flux)
    return mag

def app_mag_to_abs_mag(m, d):
    M = m - 5*np.log10(d/10)
    return M

#solar abs mag per casagrande & vandenerg 2018
# https://arxiv.org/pdf/1806.01953
def mag_to_lum(M_1, M_2=4.67, L_2=L_sun): 
    L_1 = L_2 * 10**((M_2 - M_1) / 2.5)
    return L_1