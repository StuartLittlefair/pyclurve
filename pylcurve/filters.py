import numpy as np
from os import path
from scipy.integrate import simps
import astropy.units as u
from glob import glob
from scipy.interpolate import interp1d
from collections import OrderedDict
from pkg_resources import resource_filename


class filters:
    """
    Class to hold wavelength and transmission arrays for both SDSS and
    super SDSS (HiPERCAM) filter profiles
    """
    
    def __init__(self, instrument='hcam', fpath=resource_filename('pylcurve', 'data/filter_profiles/')):

        choices = ['hcam', 'ucam', 'ucam_s', 'ucam_super', 'ucam_sloan',
                   'ucam_old', 'uspec', 'sdss']
        self.inst = instrument
        if self.inst not in choices:
            raise ValueError('"{}" is not a valid instrument'
                             .format(self.inst))
        self.fpath = fpath
        self.wl, self.trans, self.bands = self.load_bandpasses(self.fpath, self.inst)
        self.eff_wl = self.pivot_wl()


    def load_bandpasses(self, fpath, inst):
        wl = dict()
        trans = dict()

        if (inst == 'ucam_sloan') or (inst == 'ucam_old'):
            inst = 'ucam'
            fnames = [path.split(f)[1] for f in glob(fpath + 'ucam_*.txt')
                      if 's' not in path.split(f)[1]]
            bands = [path.splitext(fname)[0].split('_')[-1] for fname in fnames]
        elif inst == 'ucam' or inst == 'ucam_s' or inst == 'ucam_super':
            inst = 'ucam'
            fnames = [path.split(f)[1] for f in glob(fpath +  'ucam_*.txt')
                      if 's' in path.split(f)[1]]
            bands = [path.splitext(fname)[0].split('_')[-1] for fname in fnames]
        else:
            fnames = glob(fpath + inst + '_*.txt')
            fnames_cut = [fname.split('_')[-1] for fname in fnames]
            bands = [fname.split('.')[0] for fname in fnames_cut]
        
        for band in bands:
            file = (fpath + inst + '_{}.txt'.format(band))
            wl[band], trans[band] = np.loadtxt(file, unpack=True)
            wl[band] *= u.AA
        return wl, trans, bands


    def pivot_wl(self):
        self.eff_wl = OrderedDict()
        eff_wls = np.array([(simps(self.trans[band], self.wl[band]) 
                        / simps(self.trans[band]
                        / self.wl[band]**2, self.wl[band]))**0.5
                            for band in self.bands])
        self.bands = [x for _,x in sorted(zip(eff_wls,self.bands))]
        eff_wls = sorted(eff_wls) * u.AA  # dictionary??
        for i, band in enumerate(self.bands):
            self.eff_wl[band] = eff_wls[i]
        return self.eff_wl


    def synphot(self, wave, flux, band):
        flux = flux.to(u.Jansky, equivalencies=u.spectral_density(wave))
        trans_interpolator = interp1d(self.wl[band], self.trans[band],
                                    bounds_error=False, fill_value=0)
        trans_new = trans_interpolator(wave)
        filtered_spec = flux * trans_new
        fLam = (simps(filtered_spec, wave) / simps(trans_new, wave))
        return fLam * u.Jansky


    def synphot_mag(self, wave, flux, band):
        flux = flux.to(u.Jansky, equivalencies=u.spectral_density(wave))
        # synflux = self.Synphot(wave, flux, band)
        mag = -2.5 * np.log(flux / (3631*u.Jansky))
        return mag