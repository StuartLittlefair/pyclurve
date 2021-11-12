from astropy.io import ascii
from pkg_resources import resource_filename
from scipy.interpolate import LinearNDInterpolator, interp1d
from filters import filters
from glob import glob
x

"""
Creates interpolator for use in utils.get_gdark function.
Interpolates precomputed tables of gravity darkening coefficients

Outputs interpolator
"""

fpath = resource_filename('pylcurve', 'data/gravity_darkening_coeffs/')

hcam = filters()
sdss = filters('sdss')
gdark_interpolator = dict()

for band in hcam.bands + sdss.bands:
    fname = glob(fpath + 'MS_GDCs_{}.dat'.format(band))[0]
    gdcs = ascii.read(fname)
    coords_in = list(zip(gdcs['Teff'], gdcs['log(g)']))
    coords_out = list(gdcs['y'])
    if band in hcam.bands:
        coords_out = list(zip(gdcs['y1'], gdcs['y2']))
    gdark_interpolator[band] = LinearNDInterpolator(coords_in,
                                                    coords_out,
                                                    rescale=True)

beta_dat = ascii.read(fpath + 'beta.dat')
beta_interpolator = interp1d(beta_dat['logTeff'], beta_dat['Beta1'])