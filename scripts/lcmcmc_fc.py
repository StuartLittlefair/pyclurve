import emcee
from multiprocessing import Pool
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pylcurve.lcurve import Lcurve
from pylcurve.modelling import Model
import pylcurve.mcmc_utils as m
import pylcurve.utils as utils
from pylcurve.filters import filters


"""
This script fits flux calibrated, multi-band primary eclipse photometry of
WD-WD/WD-dM binaries using an MCMC method to run Tom Marsh's LROCHE routine.
Using mass-radius relations it can determine stellar masses and effective
temperatures for both components.

All typical user modified variables are denoted by "ENTER" for ease.
"""


# ENTER filter system of observations

# cam = filters('sdss') ## SDSS throughputs
# cam = filters('ucam_sloan') ## ULTRACAM throughputs with standard sloan filters
cam = filters('ucam') ## ULTRACAM throughputs with super filters
# cam = filters('hcam') ## HiPERCAM throughputs with super filters


class EclipseLC(Model):
    parameter_names = ('t1', 't2', 'm1', 'm2', 'incl', 't0', 'per', 'parallax')
    def __init__(self, model_file, lightcurves, *args, **kwargs):
        """
        A lightcurve model for an eclipsing WD-WD/WD-dM.

        Parameters
        ----------
        model_file: model containing LCURVE file with auxillary (fixed) params
        lightcurves: a dictionary of band: filename pairs

        The remaining parameters are either passed in as a list of arguments
        (in order) or specified as a dictionary:

        t1, t2 :  white dwarf/M-dwarf temp in K
        m1, m2 : white dwarf/M-dwarf masses in solar masses
        incl : inclination of system
        t0 : mid-eclipse time of primary eclipse
        per : orbital period of system
        parallax : parallax of system
        """
        super().__init__(*args, **kwargs)
        self.lightcurves = lightcurves
        self.model_file = model_file


    def get_value(self, band):
        """
        Calculate lightcurve

        Parameters
        ----------
        band : string
            SDSS/HiPERCAM bandpass

        Returns
        -------
        ym : np.ndarray
            model values
        """
        # setup LCURVE file for this band
        lcurve_model = Lcurve(self.model_file)
        lcurve_pars = dict()
        q = self.m2/self.m1

        # ENTER chosen mass-radius relations for both stars
        self.r1 = utils.get_radius(self.m1, self.t1, star_type='CO')
        self.r2 = utils.get_radius(self.m2, self.t2, star_type='MS')

        log_g1 = utils.log_g(self.m1, self.r1)
        log_g2 = utils.log_g(self.m2, self.r2)
        a = utils.separation(self.m1, self.m2, self.per)

        # ENTER interstellar redenning/extinction
        ebv = 0.05
        Av = 3.1 * ebv

        scale_factor = utils.scalefactor(a, self.parallax, cam.eff_wl[band], Av)
        lcurve_pars['t1'] = utils.get_Tbb(self.t1, log_g1, band, star_type='WD',
                                          source='Bergeron')
        lcurve_pars['t2'] = utils.get_Tbb(self.t2, log_g2, band, star_type='MS')
        lcurve_pars['r1'] = self.r1/a  # scale to separation units
        lcurve_pars['r2'] = utils.Rva_to_Rl1(q, self.r2/a)  # scale and correct
        lcurve_pars['t0'] = self.t0
        lcurve_pars['period'] = self.per
        lcurve_pars['iangle'] = self.incl
        lcurve_pars['q'] = q
        lcurve_pars['wavelength'] = cam.eff_wl[band].to_value(u.nm)
        lcurve_pars['phase1'] = np.arcsin(lcurve_pars['r1'] - lcurve_pars['r2']) / (2 * np.pi)
        lcurve_pars['phase2'] = 0.5 - lcurve_pars['phase1']
        lcurve_model.set(lcurve_pars)
        lcurve_model.set(utils.get_ldcs(self.t1, logg_1=log_g1, band=band,
                                        star_type_1='WD', teff_2=self.t2,
                                        logg_2=log_g2, star_type_2='MS'))
        
        if not lcurve_model.ok():
            raise ValueError('invalid parameter combination')
        _, _, _, ym = lcurve_model(self.lightcurves[band], scale_factor)
        return ym

    def log_prior(self):
        """
        Prior probabilities
        """
        # first call parent class log_prior -> checks params in bounds
        val = super().log_prior()
        if np.isinf(val):
            return val

        # ENTER prior on parallax from Gaia (NN Ser)
        par_prior = m.Prior('gaussPos', 1.9166, 0.0980)
        val += par_prior.ln_prob(self.parallax)
        
        # ENTER prior on T0
        prior = m.Prior('gauss', 55307.400302182999, 1.3524578e-06)
        val += prior.ln_prob(self.t0)

        # ENTER prior on period
        prior = m.Prior('gauss', 0.13008017141, 0.00000000017)
        val += prior.ln_prob(self.per)

        return val

    def plot(self, ax, band, params, style='whole', dcolor='k'):
        """
        Plots data and model on axis

        style is either whole, model or residuals.
            'whole' plots the raw data and the full model (mean model + GP).

            'model' plots the mean model and the data after subtraction
             of the mean of the GP - i.e the data minus the pulsations

            'residuals' plots the data minus the mean model, together with the
            mean and range of the GP
        """
        self.set_parameter_vector(params)
        t, _, y, ye, _, _ = np.loadtxt(self.lightcurves[band]).T
        ym = self.get_value(band)

        toff = int(np.floor(np.min(t)))
        tplot = t - toff

        if style == 'whole':
            ax.errorbar(tplot, y, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.plot(tplot, ym, color='k', lw=2, ls=':')
        elif style == 'residuals':
            ax.errorbar(tplot, y-ym, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
        else:
            raise ValueError('style not recognised')

    def log_probability(self, params, band):
        """
        Calculate log of posterior probability

        Parameters
        -----------
        params : iterable
            list of parameter values
        band : string
            SDSS/HiPERCAM band
        """
        self.set_parameter_vector(params)
        _, _, y, ye, _, _ = np.loadtxt(self.lightcurves[band]).T

        # check model params are valid - checks against bounds
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        try:
            ym = self.get_value(band)
        except ValueError as err:
            # invalid lcurve params
            print('warning: model failed ', err)
            return -np.inf
        else:
            chisq = np.sum((y - ym)**2 / ye**2)
            log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * ye**2)) + chisq)
            # log_likelihood + lp
            return log_likelihood


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fit or plot model of LC')
    parser.add_argument('--nwalkers', action='store', type=int, default=50)
    parser.add_argument('--fit', '-f', action='store_true')
    parser.add_argument('--nburn', action='store', type=int, default=1000)
    parser.add_argument('--nprod', action='store', type=int, default=10000)
    parser.add_argument('--nthreads', action='store', type=int, default=1)
    args = parser.parse_args()

    ###########################################################################

    nameList = np.array(['t1', 't2', 'm1', 'm2', 'incl', 't0', 'per', 'parallax'])
    
    # ENTER starting parameters (in same order as namelist)
    params = np.array([57000, 3300, 0.535, 0.111, 89.6, 55307.400302182999,
                       0.13008017141, 1.9166])
    ndim = len(params)

    # ENTER model limits
    model_bounds = dict(
        t1=(5000, 80000),
        t2=(2300, 6900),
        m1=(0.35, 9),
        m2=(0.06, 0.7),
        incl=(70, 90),
        t0=(55307.4000, 55307.4005),
        per=(0.13007967141, 0.13008067141),
        parallax=(1, 3)
    )
    # ENTER light curve data
    lc_path = 'light_curves/'
    light_curves = dict(
        u=lc_path + 'NN_Ser_ucam_sloan_uband_cut_fc.dat',
        g=lc_path + 'NN_Ser_ucam_sloan_gband_cut_fc.dat',
        i=lc_path + 'NN_Ser_ucam_sloan_iband_cut_fc.dat',
    )
    # ENTER lcurve model filename
    model_file = 'model'

    ###########################################################################
    model = EclipseLC(model_file, light_curves, *params, bounds=model_bounds)

    # wrapper to combine log probability from all bands
    def log_probability(params):
        val = 0
        for band in light_curves.keys():
            val += model.log_probability(params, band)
        model.set_parameter_vector(params)
        val += model.log_prior()
        return val

    if args.fit:
        nwalkers = args.nwalkers

        def log_prior(params):
            model.set_parameter_vector(params)
            return model.log_prior()

        # amount to scatter initial ball of walkers
        scatter = 0.01*np.ones_like(params)
        # small scatter for t0 and period
        scatter[5] = 1.0e-9
        scatter[6] = 1.0e-9
        pool = Pool(args.nthreads)
        p0 = m.initialise_walkers(params, scatter, nwalkers, log_prior)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

        pos, prob, state = m.run_burnin(sampler, p0, args.nburn)
        sampler.reset()

        sampler = m.run_mcmc_save(sampler, pos, args.nprod, state, 'chain.txt')
        chain = m.flatchain(sampler.chain, ndim, thin=3)

        bestPars = []
        for i in range(ndim):
            par = chain[:, i]
            lolim, best, uplim = np.percentile(par, [16, 50, 84])
            print("%s = %f +%f -%f" % (nameList[i], best, uplim-best, best-lolim))
            bestPars.append(best)
        fig = m.thumbPlot(chain, nameList)
        fig.savefig('cornerPlot.pdf')
        plt.close()

        gs = gridspec.GridSpec(3, 2)
        gs.update(hspace=0.0)

        shared_ax = None
        for iband, band in enumerate(light_curves.keys()):###################
            if shared_ax:
                ax_main = plt.subplot(gs[iband, 0], sharex=shared_ax)
                ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
            else:
                ax_main = plt.subplot(gs[iband, 0])
                shared_ax = ax_main
                ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
            color = sns.color_palette('nipy_spectral', 3)[iband-1]

            model.plot(ax_main, band, bestPars, style='whole', dcolor=color)
            model.plot(ax_res, band, bestPars, style='residuals', dcolor=color)
            if band != list(light_curves.keys())[-1]:
                plt.setp(ax_main.get_xticklabels(), visible=False)
                plt.setp(ax_res.get_xticklabels(), visible=False)
        plt.savefig('lightCurves.pdf')
        plt.close()

    else:
        try:
            chain = m.readchain('chain.txt')
            fchain = m.flatchain(chain, ndim+1, thin=3)[:, :-1]
            bestPars = np.median(fchain, axis=0)
            lolim, uplim = np.percentile(fchain, (16, 84), axis=0)
            for name, par, lo, hi in zip(nameList, bestPars, lolim, uplim):
                print('{} = {} + {} - {}'.format(name, par, hi-par, par-lo))

            fig = m.thumbPlot(fchain, nameList)
            fig.savefig('cornerPlot.pdf')
            plt.close()

            gs = gridspec.GridSpec(3, 2)
            gs.update(hspace=0.0)

            shared_ax = None
            for iband, band in enumerate(light_curves.keys()):###################
                if shared_ax:
                    ax_main = plt.subplot(gs[iband, 0], sharex=shared_ax)
                    ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
                else:
                    ax_main = plt.subplot(gs[iband, 0])
                    shared_ax = ax_main
                    ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
                color = sns.color_palette('nipy_spectral', 3)[iband-1]

                model.plot(ax_main, band, bestPars, style='whole', dcolor=color)
                model.plot(ax_res, band, bestPars, style='residuals', dcolor=color)
                if band != list(light_curves.keys())[-1]:
                    plt.setp(ax_main.get_xticklabels(), visible=False)
                    plt.setp(ax_res.get_xticklabels(), visible=False)
            plt.savefig('lightCurves.pdf')
            plt.show()
            plt.close()

        except Exception as err:
            print('no chain read, falling back to guess ' + str(err))

            print('Best fit has ln_prob of {}'.format(log_probability(params)))

            gs = gridspec.GridSpec(3, 2)
            gs.update(hspace=0.0)

            shared_ax = None
            for iband, band in enumerate(light_curves.keys()):###################
                if shared_ax:
                    ax_main = plt.subplot(gs[iband, 0], sharex=shared_ax)
                    ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
                else:
                    ax_main = plt.subplot(gs[iband, 0])
                    shared_ax = ax_main
                    ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
                color = sns.color_palette('nipy_spectral', 3)[iband-1]

                model.plot(ax_main, band, params, style='whole', dcolor=color)
                model.plot(ax_res, band, params, style='residuals', dcolor=color)
                if band != list(light_curves.keys())[-1]:
                    plt.setp(ax_main.get_xticklabels(), visible=False)
                    plt.setp(ax_res.get_xticklabels(), visible=False)
            plt.savefig('lightCurves_nochain.pdf')
            plt.show()
            plt.close()
