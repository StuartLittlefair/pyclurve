from astropy.table import Table
from pkg_resources import resource_filename
from scipy.interpolate import LinearNDInterpolator
from .filters import filters
from glob import glob


"""
Creates interpolators for use in utils.get_ldcs function.
Interpolates precomputed tables of limb darkening coefficients for 2 stars
(some combination of white dwarf and/or main-sequence stars).
Claret (2000) 4-parameter law only.

Outputs interpolators as dictionary
>>> ld_interpolator[star_type][band](Teff, log(g))
"""

hcam = filters('hcam')
sdss = filters('sdss')


fpath = resource_filename('pylcurve', 'data/ld_coeffs/')
ld_wd_interpolators = dict()
ld_ms_interpolators = dict()
ld_interpolator = dict()
for band in hcam.bands + sdss.bands:
    filename_wd = glob(fpath + 'DA_LDCs_*_{}.dat'.format(band))[0]
    tab_wd = Table.read(filename_wd, format='ascii')
    wd_coords_in = list(zip(tab_wd['Teff'], tab_wd['log(g)']))
    wd_coords_out = list(zip(tab_wd['a1'], tab_wd['a2'],
                             tab_wd['a3'], tab_wd['a4']))
    ld_wd_interpolators[band] = LinearNDInterpolator(wd_coords_in,
                                                     wd_coords_out,
                                                     rescale=True)

    filename_ms = glob(fpath + 'MS_LDCs_*_{}.dat'.format(band))[0]
    tab_ms = Table.read(filename_ms, format='ascii')
    ms_coords_in = list(zip(tab_ms['Teff'], tab_ms['log(g)']))
    ms_coords_out = list(zip(tab_ms['a1'], tab_ms['a2'],
                             tab_ms['a3'], tab_ms['a4']))
    ld_ms_interpolators[band] = LinearNDInterpolator(ms_coords_in,
                                                     ms_coords_out,
                                                     rescale=True)

ld_interpolator['WD'] = ld_wd_interpolators
ld_interpolator['MS'] = ld_ms_interpolators