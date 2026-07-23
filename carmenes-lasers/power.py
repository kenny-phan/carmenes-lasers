import astropy.units as u
import numpy as np

from astropy.constants import L_sun
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

def gaia_query(ra, dec, radius=5*u.arcsec):
    coords = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    gaia_result = Gaia.query_object(coordinate=coords, radius=radius)
    return gaia_result

def parallax_to_distance(parallax): # in arcsec
    distance = (1 / parallax.value)
    return distance * u.parsec

def plax_err_to_dist_err(parallax, parallax_err):
    dist_err = (parallax_err.value / parallax.value**2)
    return dist_err * u.parsec

# per: 
# https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html
# Evans+ 2018 DR2 
# https://arxiv.org/pdf/1804.09368
def flux_to_app_mag(flux, zp=25.6884): # Gaia G-Band zero point
    mag = zp - 2.5*np.log10(flux)
    return mag

def app_mag_to_abs_mag(m, d):
    M = m - 5*np.log10(d/10)
    return M

#solar abs mag per casagrande & vandenerg 2018
# https://arxiv.org/pdf/1806.01953
def abs_mag_to_lum(M_1, M_2=4.67, L_2=L_sun): 
    L_1 = L_2 * 10**((M_2 - M_1) / 2.5)
    return L_1

def flux_to_app_mag_err(flux, flux_err, zp_err=0.0018):
    dmdz = 1
    dmdflux = -2.5*(1/flux * np.log(10))

    zp_comp = (dmdz*zp_err)**2
    flux_comp = (dmdflux*flux_err)**2
    
    m_err = np.sqrt(zp_comp + flux_comp)
    return m_err

#eqn: M = m - 5*log10(d/10)
def app_mag_to_abs_mag_err(d, d_err, m_err):
    dMdm = 1
    dmdd = -5*(10/d * np.log(10))

    m_comp = (dMdm*m_err)**2
    d_comp = (dmdd*d_err)**2

    M_err = np.sqrt(m_comp + d_comp)
    return M_err

#eqn: L_1 = L_2 * 10**((M_2 - M_1)/2.5)
def abs_mag_to_lum_err(M_1, M_1_err, 
                       M_2=4.67, M_2_err=0.01, 
                       L_2=L_sun.value, L_2_err=0.0):

    dL_1dM_1 = L_2 * ((M_2 - M_1)/2.5) * 10**((M_2 - M_1)/2.5 - 1) * (-1/2.5)
    dL_1dM_2 = L_2 * ((M_2 - M_1)/2.5) * 10**((M_2 - M_1)/2.5 - 1) * (1/2.5)
    dL_1dL_2 = 10**((M_2 - M_1)/2.5)

    M_1_comp = (dL_1dM_1*M_1_err)**2
    M_2_comp = (dL_1dM_2*M_2_err)**2
    L_2_comp = (dL_1dL_2*L_2_err)**2

    L_1_err = np.sqrt(M_1_comp + dL_1dM_2 + L_2_comp)
    return L_1_err
    

def gaia_to_lum(result):
    dist_arcsec = (result['parallax']).to(u.arcsec)
    dist_arcsec_err = (result['parallax_error']).to(u.arcsec)
    
    flux = result['phot_g_mean_flux'] # electrons / sec
    flux_err = result['phot_g_mean_flux_error'] 
        
    app_mag = flux_to_app_mag(flux.value)
    app_mag_err = flux_to_app_mag_err(flux.value, flux_err.value)
    
    dist = parallax_to_distance(dist_arcsec)
    dist_err = plax_err_to_dist_err(dist_arcsec, dist_arcsec_err)
    
    abs_mag = app_mag_to_abs_mag(app_mag, dist.value)
    abs_mag_err = app_mag_to_abs_mag_err(dist, dist_err, app_mag_err)
    
    lum = abs_mag_to_lum(abs_mag)
    lum_err = abs_mag_to_lum_err(abs_mag, abs_mag_err)*u.W
    
    return lum, lum_err