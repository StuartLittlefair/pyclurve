import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d, LinearNDInterpolator
from pkg_resources import resource_filename


mr_interpolator = dict()
ms_interpolator = dict()

fpath = resource_filename('pylcurve', 'data/cooling_tracks/')

# he = ascii.read(fpath + 'He_tracks_thick.dat')
he = ascii.read(fpath + 'He_tracks.dat')
co = ascii.read(fpath + 'CO_tracks.dat')
one = ascii.read(fpath + 'ONe_tracks.dat')
mass, radius = np.loadtxt(fpath + 'MdwarfMRrel.dat', unpack=True)
std_mass, std_radius = np.loadtxt(fpath + 'Mdwarf_stds.dat', unpack=True)
baraffe = ascii.read(fpath + 'Baraffe/baraffe.dat')

he_coords_in = list(zip(he['M'], he['Teff']))
he_coords_out = list(he['R'])

co_coords_in = list(zip(co['M'], co['Teff']))
co_coords_out = list(co['R'])

one_coords_in = list(zip(one['M'], one['Teff']))
one_coords_out = list(one['R'])

coords_in = list(zip(baraffe['M/Ms'], 10**(baraffe['logt(yr)']-9)))
coords_out = list(baraffe['R/Rs'])

he_interpolator = LinearNDInterpolator(he_coords_in, he_coords_out,
                                       rescale=True)
co_interpolator = LinearNDInterpolator(co_coords_in, co_coords_out,
                                       rescale=True)
one_interpolator = LinearNDInterpolator(one_coords_in, one_coords_out,
                                        rescale=True)
ms_interpolator['baraffe'] = LinearNDInterpolator(coords_in, coords_out, rescale=True)
ms_interpolator['empirical'] = interp1d(mass, radius)
ms_interpolator['std'] = interp1d(std_mass, std_radius)

mr_interpolator['He'] = he_interpolator
mr_interpolator['CO'] = co_interpolator
mr_interpolator['ONe'] = one_interpolator
mr_interpolator['MS'] = ms_interpolator