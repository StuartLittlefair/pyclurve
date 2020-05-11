import numpy as np
import astropy.units as u
from scipy.integrate import quad
from astropy.constants import G
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from .blackbody import bb_interpolator
from .limbdark import ld_interpolator
from .massradius import mr_interpolator
from .rochedistortion import roche_interpolator
from dust_extinction.parameter_averages import F20


def get_Tbb(teff, logg, band, instrument='ucam_sloan',
            star_type='WD', source='Bergeron'):
    """
    Interpolates Teff to Tbb tables for a given filter band
    ('u', 'g', 'r', 'i', or 'z') for PHOENIX main-sequence
    models (star_type='MS') or Bergeron & Gianninas white dwarf
    models (star_type='WD').
    """
    if star_type == 'WD' and source == 'Claret':
        T_bb = float(bb_interpolator['WD_Claret'][instrument][band](teff, logg))
    elif star_type == 'WD' and source == 'Bergeron':
        T_bb = float(bb_interpolator['WD'][instrument][band](teff, logg))
    else:
        T_bb = float(bb_interpolator[star_type][instrument][band](teff, logg))
    return T_bb


def get_ldcs(teff_1, logg_1, band, star_type_1='WD',
             teff_2=None, logg_2=None, star_type_2=None):
    """
    Interpolates Claret WD (star_type='WD') and PHOENIX MS (star_type='MS')
    tables and returns a dictionary of limb-darkening coefficients for a given
    filter band ('u', 'us', 'g', 'gs', 'r', 'rs', 'i', 'is', 'z', or 'zs').
    """

    ldcs = dict()

    ldcs['ldc1_1'], ldcs['ldc1_2'], ldcs['ldc1_3'], ldcs['ldc1_4'] = (
        ld_interpolator[star_type_1][band](teff_1, logg_1)
    )
    if star_type_2:
        ldcs['ldc2_1'], ldcs['ldc2_2'], ldcs['ldc2_3'], ldcs['ldc2_4'] = (
            ld_interpolator[star_type_2][band](teff_2, logg_2)
        )
    return ldcs


def get_radius(mass, temp=None, star_type='CO',
               relation='empirical', age_gyr=5):
    """
    Interpolates mass-radius relations for WDs (CO-core('CO') or He-core('He'))
    and for M-dwarfs ('empirical' or 'baraffe'). For baraffe, tracks of
    different ages can be selected.
    """
    if star_type=='He' or star_type=='CO':
        radius = mr_interpolator[star_type](mass, temp)
    elif star_type=='MS':
        if relation=='empirical':
            radius = mr_interpolator[star_type][relation](mass)
        else:
            radius = mr_interpolator[star_type][relation](mass, age_gyr)
    return radius


def Rva_to_Rl1(q, r_VA_a):
    """
    Correct scaled volume-averaged radius to roche distorted radius towards L1
    given the mass ratio (M2/M1).
    """
    r_L1_a = roche_interpolator(q, r_VA_a)
    return r_L1_a


def log_g(m, r):
    """
    Calculate log(g) [cgs] given Mass and Radius in solar units.
    """
    m = m*u.M_sun
    r = r*u.R_sun
    return np.log10((G*m/r/r).to_value(u.cm/u.s/u.s))


def separation(m1, m2, p):
    """
    Calculation binary separation
    Parameters
    -----------
    m1 : float
        mass of star 1 in solar units
    m2 : float
        mass of star 2 in solar units
    p : float
        period in days
    Returns
    -------
    a : float
        binary separation in solar radii
    """
    mt = (m1+m2) * u.M_sun
    p = p * u.d
    acubed = G * p**2 * mt / 4 / np.pi**2
    a = acubed ** (1/3)
    return a.to_value(u.R_sun)


def Claret_LD_law(mu, c1, c2, c3, c4):
    """
    Claret 4-parameter limb-darkening law.
    """
    I = (1 - c1*(1 - mu**0.5) - c2*(1 - mu) - c3*(1 - mu**1.5) - c4*(1 - mu**2)) * mu
    return I


def m1m2(vel_scale, q, P):
    vel_scale = vel_scale * (u.km / u.s)
    P = P * u.d
    m1 = ((P * vel_scale**3) / (2 * np.pi * G * (1 + q**-1))).to_value(u.M_sun)
    m2 = ((P * vel_scale**3) / (2 * np.pi * G * (1 + q))).to_value(u.M_sun)
    return m1, m2


def scalefactor(a, parallax, wavelength=550*u.nm, Av=0):
    """
    Calculates an Lcurve scalefactor for use with flux calibrated data given
    an orbital separation and parallax, taking reddening into account.
    For data with flux units of Janskys.
    """
    a = a * u.R_sun
    # parallax (mas) to distance (parsecs)
    d = ((1000 / parallax) * u.parsec).to(u.R_sun)
    # lcurve flux is in W/m^2/separation^2 -> correct to Janskys
    # and account for reddening
    ext = F20(Rv=3.1)
    extinction_fac = ext.extinguish(wavelength, Av)
    sf = (a**2 / d**2) * extinction_fac
    return sf * 10**26