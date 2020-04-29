import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d, LinearNDInterpolator
from pkg_resources import resource_filename


mr_interpolator = dict()
ms_interpolator = dict()

fpath = resource_filename('pylcurve', 'data/cooling_tracks/')

he = ascii.read(fpath + 'He_tracks_thick.dat')
co = ascii.read(fpath + 'CO_tracks.dat')
mass, radius = np.loadtxt(fpath + 'MdwarfMRrel.dat', unpack=True)
baraffe = ascii.read(fpath + 'Baraffe/baraffe.dat')

he_coords_in = list(zip(he['Mass'], he['Teff']))
he_coords_out = list(he['Radius'])

co_coords_in = list(zip(co['M'], co['Teff']))
co_coords_out = list(co['R'])

coords_in = list(zip(baraffe['M/Ms'], 10**(baraffe['logt(yr)']-9)))
coords_out = list(baraffe['R/Rs'])

he_interpolator = LinearNDInterpolator(he_coords_in, he_coords_out,
                                        rescale=True)
co_interpolator = LinearNDInterpolator(co_coords_in, co_coords_out,
                                       rescale=True)
ms_interpolator['baraffe'] = LinearNDInterpolator(coords_in, coords_out, rescale=True)
ms_interpolator['empirical'] = interp1d(mass, radius)

mr_interpolator['He'] = he_interpolator
mr_interpolator['CO'] = co_interpolator
mr_interpolator['MS'] = ms_interpolator