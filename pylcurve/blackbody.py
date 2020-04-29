from astropy.table import Table
import pkg_resources
from scipy.interpolate import LinearNDInterpolator

bands = ('u', 'g', 'r', 'i', 'z')
blackbody_interpolators = dict()
filename = 'data/Twd2Tbb.dat'
filename = pkg_resources.resource_filename('pylcurve', filename)
t = Table.read(filename, format='ascii.tab')

T = t['Teff']
G = t['logg']

for band in bands:
    temps = t['T_BB_' + band]

    coords_in = list(zip(T, G))
    Tbb_coords_out = list(temps)
    blackbody_interpolators[band] = LinearNDInterpolator(coords_in, Tbb_coords_out)