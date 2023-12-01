import numpy as np
import astropy.units as u
from scipy.integrate import quad, simps
from astropy.constants import G, sigma_sb, c, h, k_B
from astropy.modeling.physical_models import BlackBody
from .filters import filters
from .blackbody import bb_interpolator
from .limbdark import ld_interpolator
from .gravitydark import gdark_interpolator, beta_interpolator
from .massradius import mr_interpolator
from .rochedistortion import roche_interpolator
from trm import roche
from dust_extinction.parameter_averages import F19

bb = BlackBody()
ext = F19(Rv=3.1)
hcam = filters('hcam')
ucam = filters('ucam')
ucam_sloan = filters('ucam_sloan')
sdss = filters('sdss')
sol = c.value
h_plnk = h.value
k_b = k_B.value

filt_dict = dict(
    hcam=hcam,
    ucam=ucam,
    ucam_sloan=ucam_sloan,
    sdss=sdss
)


def planck(wl, temp):
    const = ((2 * h_plnk * sol) / (wl * 10 ** -10) ** 3)
    frac = 1 / (np.exp((h_plnk * sol) / ((wl*10 ** -10) * temp * k_b)) - 1)
    return const * frac / 10 ** -26


def get_Tbb(teff, logg, band, instrument='ucam', star_type='WD', model='Claret'):
    """
    Interpolates Teff to Tbb tables for a given filter band
    ('u', 'g', 'r', 'i', or 'z') for PHOENIX main-sequence
    models (star_type='MS') or Bergeron & Gianninas white dwarf
    models (star_type='WD').
    """
    # if star_type == 'WD' and source == 'Claret':
    #     T_bb = float(bb_interpolator['WD']['Claret'][band](teff, logg))
    # elif star_type == 'WD' and source == 'Bergeron':
    #     T_bb = float(bb_interpolator['WD'][instrument][band](teff, logg))
    # elif star_type == 'MS' and source == 'BT-SETTL':
    #     T_bb = float(bb_interpolator['MS']['BT-SETTL'][band](teff, logg))
    # else:
    #     T_bb = float(bb_interpolator[star_type][instrument][band](teff, logg))

    T_bb = float(bb_interpolator[star_type][model][instrument][band](teff, logg))
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
        if teff_2 < 2300:
            teff_2 = 2300
        ldcs['ldc2_1'], ldcs['ldc2_2'], ldcs['ldc2_3'], ldcs['ldc2_4'] = (
            ld_interpolator[star_type_2][band](teff_2, logg_2)
        )
    return ldcs


def get_gdc(teff, logg, band, beta=None):
    if not beta:
        if teff < 2000: teff = 2000
        beta = beta_interpolator(np.log10(teff))
    if teff < 2300: teff = 2300
    if band in hcam.bands:
        return beta * gdark_interpolator[band](teff, logg)[0] + gdark_interpolator[band](teff, logg)[1]
    else:
        return beta * gdark_interpolator[band+'s'](teff, logg)[0] + gdark_interpolator[band+'s'](teff, logg)[1]


def get_radius(mass, temp=None, star_type='CO',
               relation='empirical', factor=1.0, age_gyr=5):
    """
    Interpolates mass-radius relations for WDs (CO-core('CO') or He-core('He'))
    and for M-dwarfs ('empirical' or 'baraffe'). For baraffe, tracks of
    different ages can be selected.
    """
    if star_type=='He' or star_type=='CO' or star_type=='ONe':
        radius = mr_interpolator[star_type](mass, temp)
    elif star_type=='MS':
        if relation=='empirical':
            radius = mr_interpolator[star_type][relation](mass)
        else:
            radius = mr_interpolator[star_type][relation](mass, age_gyr)
    return radius * factor


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


def separation(m1, m2, P):
    """
    Calculates binary separation
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
    P = P * u.d
    acubed = G * P**2 * mt / 4 / np.pi**2
    a = acubed ** (1/3)
    return a.to_value(u.R_sun)


def rv_semiamplitudes(m1, m2, P, i):
    """
    Calculates radial velocity semiamplitudes.
    Parameters
    -----------
    m1 : float
        mass of star 1 in solar units
    m2 : float
        mass of star 2 in solar units
    P : float
        period in days
    i : float
        inclination in degrees
    Returns
    -------
    k1 : float
        radial velocity semiamplitude of star 1 in km/s
    k2 : float
        radial velocity semiamplitude of star 2 in km/s
    """
    m = np.array([m2, m1]) * u.M_sun
    m1 = m1*u.M_sun
    m2 = m2*u.M_sun
    P = P*u.d
    const = (2 * np.pi * G * np.sin(i*u.deg)**3) / (P * (m1 + m2)**2)
    k1, k2 = ((const * m**3)**(1/3)).to_value(u.km/u.s)
    return k1, k2


def m1m2(vel_scale, q, P):
    vel_scale = vel_scale * (u.km / u.s)
    P = P * u.d
    m1 = ((P * vel_scale**3) / (2 * np.pi * G * (1 + q**-1))).to_value(u.M_sun)
    m2 = ((P * vel_scale**3) / (2 * np.pi * G * (1 + q))).to_value(u.M_sun)
    return m1, m2


def t2phase(t, t0, P):
    phase = ((t - t0) / P) % 1
    phase[phase > 0.5] -=1 
    return phase


def contact_points(t0, P, q, incl, r1_a, r2_a, ntheta=100):
    try:
        c3, c4 = roche.wdphases(q, incl, r1_a, r2_a, ntheta=100)
    except roche.RocheError:
        return None, None, None, None
    t1 = t0 - c4*P
    t2 = t0 - c3*P
    t3 = t0 + c3*P
    t4 = t0 + c4*P
    if c3 == -1:
        return t1, None, None, t4
    else:
        return t1, t2, t3, t4
    

def Claret_LD_law(mu, c1, c2, c3, c4):
    """
    Claret 4-parameter limb-darkening law.
    """
    I = (1 - c1*(1 - mu**0.5) - c2*(1 - mu) - c3*(1 - mu**1.5) - c4*(1 - mu**2)) * mu
    return I


def scalefactor(a, parallax, wavelength=550*u.nm, Ebv=0):
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
    ext = F19(Rv=3.1)
    extinction_fac = ext.extinguish(wavelength, Ebv)
    sf = (a**2 / d**2) * 10**26 * extinction_fac
    return sf


def integrate_disk(teff, logg, radius, parallax, Ebv, band, wd_model='Claret', instrument='ucam'):
    # if wd_model != 'Claret':
    #     cam = filters(wd_model)
    # else:
    # cam = filters(instrument)
    cam = filt_dict[instrument]

    # ext = F19(Rv=3.1)
    # Central intensity in Janskys from blackbody temperature for model
    t_bb = float(bb_interpolator['WD'][wd_model][instrument][band](teff, logg))
    # I_cen = (bb.evaluate(cam.eff_wl[band], t_bb*u.K, scale=1) * u.sr).to_value(u.Jansky)
    I_cen = planck(cam.eff_wl[band].value, t_bb)
    # Get limb darkening for model
    c1, c2, c3, c4 = ld_interpolator['WD'][band](teff, logg)
    # Integrate total flux of disk from central intensity and limb-darkening law
    int_flux = 2 * np.pi * I_cen * quad(Claret_LD_law, 0, 1, args=(c1, c2, c3, c4))[0]
    # Scale to distance and account for extinction
    d = ((1000 / parallax) * u.parsec).to(u.R_sun)
    R = (radius * u.R_sun)
    scale =  (radius**2/d**2).value * ext.extinguish(cam.eff_wl[band], Ebv=Ebv)
    flux = scale * int_flux
    return flux


def h(theta, r2, a):
    """
    Equation 54 from Ritter (2000). Scales irradiating flux as seen at centre of
    secondary to an element on its surface at angle theta from centre-line
    between stars (assumes irradiating source is a point source as in Fig. 5.).
    """
    fs = r2/a
    top = np.cos(theta) - fs
    bottom = (1 - 2*fs*np.cos(theta) + fs**2)**1.5
    return top / bottom


def Gsin(theta, t1, r1, t2, r2, a):
    """
    Integrand of Equation 60 from Ritter (2000). Modified so G=0 for a surface
    element when F_irr >= F_0.
    """
    F_irr = (sigma_sb.value * r1**2 * t1**4 * h(theta, r2, a)) / a**2
    F_0 = sigma_sb.value * t2**4
    if F_irr >= F_0:
        return 0
    else:
        return (1 - ( F_irr / F_0)) * np.sin(theta)


def irradiate(t1, r1, t2, r2, a):
    """
    Calculates inflation due to irradiation according to Ritter (2000)
    https://ui.adsabs.harvard.edu/abs/2000A%26A...360..969R/abstract.
    """
    theta_max = np.arccos(r2/a)

    # effective surface area following Eqn.60
    s_eff = 0.5 * (1 - (r2/a) - quad(Gsin, 0, theta_max, args=(t1, r1, t2, r2, a))[0])
    # inflation according to equation 17 (Ritter 2000)
    r_irr = r2 * (1 - s_eff)**-0.1
    return r_irr