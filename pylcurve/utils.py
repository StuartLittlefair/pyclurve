import numpy as np

from astropy import units as u
from astropy.constants import G

from .limbdark import tables


def log_g(m, r):
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


def get_limbdark_params(t1, g1, t2, g2, band):
    """
    Interpolates Gianninas table and returns limb-darkened params.

    Uses quadratic limb darkening law
    """
    pars = dict()
    pars['ldc1_1'] = 0.46
    pars['ldc1_2'] = 0.35
    pars['ldc2_1'] = 0.48
    pars['ldc2_2'] = 0.35
    return pars


def wd_to_bb_temp(band, t, logg):
    return t
