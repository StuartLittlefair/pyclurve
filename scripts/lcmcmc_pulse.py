#!/usr/bin/env python
import emcee
import celerite
from celerite import terms

import numpy as np
import astropy.units as u
from astropy.modeling.blackbody import blackbody_lambda

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pylcurve.lcurve import Lcurve
from pylcurve.modeling import Model
import pylcurve.mcmc_utils as m
from pylcurve.utils import log_g, separation, get_limbdark_params, wd_to_bb_temp

band_wavs = dict(
    u=3543*u.AA,
    g=4770*u.AA,
    r=6231*u.AA,
    i=7625*u.AA,
    z=9134*u.AA
)


def scale_pulsation(band, temp):
    return blackbody_lambda(band_wavs[band], temp*u.K) / blackbody_lambda(band_wavs['g'], temp*u.K)


class EclipseLC(Model):
    parameter_names = ('t1', 't2', 'm1', 'm2', 'incl', 'r1', 'r2', 't0', 'per',
                       'pulse_omega', 'pulse_q', 'pulse_temp', 'pulse_amp')

    def __init__(self, model_file, lightcurves, *args, **kwargs):
        """
        A lightcurve model for an eclipsing DWD with pulsations

        Parameters
        ----------
        model_file: model containing LCURVE file with auxillary (fixed) params
        lightcurves: a dictionary of band: filename pairs

        The remaining parameters are either passed in as a list of arguments
        (in order) or specified as a dictionary:

        t1, t2 :  white dwarf temp in K
        m1, m2 : white dwarf masses in solar masses (constrained through RV prior)
        incl : inclination of system
        r1, r2 : white dwarf radii in solar radii
        t0 : mid-eclipse time of primary eclipse
        pulse_omega : frequency of pulsations
        pulse_q : q factor of pulsations
        pulse_temp : BB temperature of pulsations
        pulse_amp : amplitude of pulsations in g band
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

        log_g1 = log_g(self.m1, self.r1)
        log_g2 = log_g(self.m2, self.r2)
        a = separation(self.m1, self.m2, self.per)
        lcurve_pars['t1'] = wd_to_bb_temp(band, self.t1, log_g1)
        lcurve_pars['t2'] = wd_to_bb_temp(band, self.t2, log_g2)
        lcurve_pars['r1'] = self.r1/a  # scale from solar radii to separation units
        lcurve_pars['r2'] = self.r2/a  # scale from solar radii to separation units
        lcurve_pars['t0'] = self.t0
        lcurve_pars['period'] = self.per
        lcurve_pars['iangle'] = self.incl
        lcurve_pars['q'] = self.m1/self.m2
        lcurve_pars['wavelength'] = band_wavs[band].to_value(u.nm)
        lcurve_model.set(lcurve_pars)
        lcurve_model.set(get_limbdark_params(self.t1, log_g1, self.t2, log_g2, band))

        if not lcurve_model.ok():
            raise ValueError('invalid parameter combination')
        x, y, e, ym = lcurve_model(self.lightcurves[band])
        return ym

    def log_prior(self):
        """
        Prior probabilities
        """
        # first call parent class log_prior -> checks params in bounds
        val = super().log_prior()
        if np.isinf(val):
            return val

        # OK, we are within limits, check M1, M2 against RV constraints
        # K1 (brighter, hotter WD rv): 186.3 +/- 1.6 km/s
        # K2 (fainter, cooler WD rv): 213.6 +/- 4.6 km/s
        # mass ratio
        q_prior = m.Prior('gauss', 1.146, 0.027)
        q_act = self.m1/self.m2
        val += q_prior.ln_prob(q_act)
        # (m1 + m2) * sini**3 = (P/2piG) * (v1+v2)**3
        mt_prior = m.Prior('gauss', 0.66, 0.02)
        val += mt_prior.ln_prob((self.m1 + self.m2) * np.sin(self.incl)**3)

        # priors on t0, p from ephemeris
        prior = m.Prior('gauss', 57460.6510218, 0.0000010)
        val += prior.ln_prob(self.t0)

        prior = m.Prior('gauss', 0.09986526542, 0.00000000010)
        val += prior.ln_prob(self.per)

        return val

    def plot(self, ax, band, params, style='whole', dcolor='k', gpcolor='r'):
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

        pulsation_amp = self.pulse_amp * scale_pulsation(band, self.pulse_temp)
        gp = self.gpdict[band]
        gp.set_parameter_vector((np.log(pulsation_amp),
                                 np.log(self.pulse_q),
                                 np.log(self.pulse_omega)))

        samples = gp.sample_conditional(y-ym, t, size=300)
        mu = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        toff = int(np.floor(np.min(t)))
        tplot = t - toff

        if style == 'whole':
            ax.errorbar(tplot, y, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.plot(tplot, ym + mu, color=gpcolor, lw=2)
            ax.plot(tplot, ym, color='k', lw=2, ls=':')
            ax.fill_between(tplot, ym+mu+std, ym+mu-std, color=gpcolor, alpha=0.6)
        elif style == 'model':
            ax.errorbar(tplot, y-mu, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.plot(tplot, ym, color=gpcolor, lw=2)
            ax.fill_between(tplot, ym+std, ym-std, color=gpcolor, alpha=0.6)
        elif style == 'residuals':
            ax.errorbar(tplot, y-ym, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.plot(tplot, mu, color=gpcolor, lw=2)
            ax.fill_between(tplot, mu+std, mu-std, color=gpcolor, alpha=0.6)
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
        t, _, y, ye, _, _ = np.loadtxt(self.lightcurves[band]).T

        # check model params are valid - checks against bounds
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf

        # make a GP for this band, if it doesnt already exist
        if not hasattr(self, 'gpdict'):
            self.gpdict = dict()

        # Oscillation params
        pulsation_amp = self.pulse_amp * scale_pulsation(band, self.pulse_temp)

        if band not in self.gpdict:
            kernel = terms.SHOTerm(np.log(pulsation_amp),
                                   np.log(self.pulse_q),
                                   np.log(self.pulse_omega))
            gp = celerite.GP(kernel)
            gp.compute(t, ye)
            self.gpdict[band] = gp
        else:
            gp = self.gpdict[band]
            gp.set_parameter_vector((
                np.log(pulsation_amp),
                np.log(self.pulse_q),
                np.log(self.pulse_omega)
            ))
            gp.compute(t, ye)

        # now add prior of Gaussian process
        lp += gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf

        try:
            ym = self.get_value(band)
        except ValueError as err:
            # invalid lcurve params
            print('warning: model failed ', err)
            return -np.inf
        else:
            return gp.log_likelihood(y - ym) + lp


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fit or plot model of LC')
    parser.add_argument('--nwalkers', action='store', type=int, default=40)
    parser.add_argument('--fit', '-f', action='store_true')
    parser.add_argument('--nburn', action='store', type=int, default=100)
    parser.add_argument('--nprod', action='store', type=int, default=100)
    parser.add_argument('--nthreads', action='store', type=int, default=4)
    args = parser.parse_args()

    nameList = np.array(['t1', 't2', 'm1', 'm2', 'incl',
                         'r1', 'r2', 't0', 'per', 'pulse_omega',
                         'pulse_q', 'pulse_temp', 'pulse_amp'])
    params = np.array([25000, 9000, 0.40, 0.35, 89.4, 0.0193, 0.0186, 57460.6510218, 0.09986526542,
                       60.0, 10, 10000, 0.005])
    ndim = len(params)

    model_bounds = dict(
        t1=(20000, 30000),
        t2=(8000, 10000),
        m1=(0.3, 0.5),
        m2=(0.3, 0.5),
        incl=(89, 90),
        r1=(0.01, 0.025), r2=(0.01, 0.025),
        t0=(57460.6508313, 57460.6512313),
        per=(0.0998650, 0.099866),
        pulse_omega=(1, 200), pulse_q=(1, 100),
        pulse_temp=(5000, 40000), pulse_amp=(0.00001, 0.1)
    )
    # dictionary of Tseries objects, one for each band
    light_curves = dict(
        u='u.dat',
        g='g.dat',
        r='r.dat',
        i='i.dat',
        z='z.dat'
    )
    model = EclipseLC('lcurve_model', light_curves, *params, bounds=model_bounds)

    # wrapper to combine log probability from all bands
    def log_probability(params):
        val = 0
        for band in ('u', 'g', 'r', 'i', 'z'):
            val += model.log_probability(params, band)
        return val

    if args.fit:
        nwalkers = args.nwalkers

        def log_prior(params):
            model.set_parameter_vector(params)
            return model.log_prior()

        # amount to scatter initial ball of walkers
        scatter = 0.01*np.ones_like(params)
        # small scatter for t0 and period
        scatter[7] = 1.0e-9
        scatter[8] = 1.0e-9
        p0 = m.initialise_walkers(params, scatter, nwalkers, log_prior)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, threads=args.nthreads)

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

    else:
        try:
            chain = m.readchain('chain.txt')
            fchain = m.flatchain(chain, ndim+1, thin=3)[:, :-1]
            bestPars = np.median(fchain, axis=0)
            lolim, uplim = np.percentile(fchain, (16, 84), axis=0)
            for name, par, lo, hi in zip(nameList, bestPars, lolim, uplim):
                print('{} = {} + {} - {}'.format(name, par, hi-par, par-lo))
        except Exception as err:
            print('no chain read, falling back to guess ' + str(err))
            bestPars = params

    print('Best fit has ln_prob of {}'.format(log_probability(bestPars)))

    gs = gridspec.GridSpec(5, 2)
    gs.update(hspace=0.0)

    shared_ax = None
    for iband, band in enumerate(('u', 'g', 'r', 'i', 'z')):
        if shared_ax:
            ax_main = plt.subplot(gs[iband, 0], sharex=shared_ax)
            ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)
        else:
            ax_main = plt.subplot(gs[iband, 0])
            shared_ax = ax_main
            ax_res = plt.subplot(gs[iband, 1], sharex=ax_main)

        color = sns.color_palette('nipy_spectral', 5)[iband-1]
        model.plot(ax_main, band, bestPars, style='whole', dcolor=color)
        model.plot(ax_res, band, bestPars, style='residuals', dcolor=color)
        if band != 'z':
            plt.setp(ax_main.get_xticklabels(), visible=False)
            plt.setp(ax_res.get_xticklabels(), visible=False)

    plt.show()
