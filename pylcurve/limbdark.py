from astropy.table import Table
import pkg_resources
from scipy.interpolate import LinearNDInterpolator

# load Gianninas 2013 tables into memory
bands = ('u', 'g', 'r', 'i', 'z')
interpolators = dict()

for band in bands:
    filename = 'data/ld_coeffs_{}.txt'.format(band)
    # get installed locations
    filename = pkg_resources.resource_filename('pylcurve', filename)
    t = Table.read(filename, format='cds')

    # data to interpolate over
    T = t['Teff']
    G = t['log(g)']
    # quadratic limb darkening params
    b = t['b']
    c = t['c']

    # should create interpolator class here so other modules can import and use
    # without overhead of creating interpolation each time
    coords_in = list(zip(T, G))
    coords_out = list(zip(b, c))
    interpolators[band] = LinearNDInterpolator(coords_in, coords_out)
