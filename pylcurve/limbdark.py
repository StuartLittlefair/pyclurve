from astropy.table import Table
import pkg_resources
from scipy.interpolate import LinearNDInterpolator

# load Gianninas 2013 tables into memory
bands = ('u', 'g', 'r', 'i', 'z')
Quad_interpolators = dict()
Claret_interpolators = dict()

for band in bands:
    filename = 'data/ld_coeffs_{}.txt'.format(band)
    # get installed locations
    # filename = pkg_resources.resource_filename('pylcurve', filename)
    t = Table.read(filename, format='cds')

    # data to interpolate over
    T = t['Teff']
    G = t['log(g)']
    # quadratic limb darkening params
    b = t['b']
    c = t['c']
    # claret limb darkening params
    c1 = t['p']
    c2 = t['q']
    c3 = t['r']
    c4 = t['s']

    # should create interpolator class here so other modules can import and use
    # without overhead of creating interpolation each time

    coords_in = list(zip(T, G))
    Quad_coords_out = list(zip(b, c))
    Claret_coords_out = list(zip(c1, c2, c3, c4))

    Quad_interpolators[band] = LinearNDInterpolator(coords_in, Quad_coords_out)
    Claret_interpolators[band] = LinearNDInterpolator(coords_in, Claret_coords_out)
