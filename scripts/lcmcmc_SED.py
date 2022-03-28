import emcee
from multiprocessing import Pool
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
from scipy.optimize import least_squares, minimize
import os
import shutil

from lcurve import Lcurve
from modelling import Model
import mcmc_utils as m
import utils as utils
from filters import filters


"""
This script fits flux calibrated, multi-band primary eclipse photometry of
WD-WD/WD-dM binaries using an MCMC method to run Tom Marsh's LROCHE routine.
Using mass-radius relations it can determine stellar masses and effective
temperatures for both components.

All typical user modified variables are denoted by "ENTER" for ease.
"""

def t2phase(t, t0, P):
    phase = ((t - t0) / P) % 1
    phase[phase > 0.5] -=1 
    return phase


class EclipseLC(Model):

    def __init__(self, config, *args, **kwargs):
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
        self.config = config
        # self.parameter_names = tuple(self.config['params'].keys())
        self.lightcurves = self.config['light_curves']
        self.cam = filters(self.config['filter_system'])
        self.flux_uncertainty = self.config['flux_uncertainty']
        self.lcurve_model = Lcurve(self.config['model_file'])
        self.set_core_parameters()
    

    def set_core_parameters(self):
        self.core_pars = dict()
        self.core_pars['period'] = self.config['period']
        self.core_pars['tperiod'] = self.config['period']
        self.core_pars['delta_phase'] = self.config['model_settings']['delta_phase']
        self.core_pars['nlat1f'] = int(self.config['model_settings']['primary_fine_resolution'])
        self.core_pars['nlat2f'] = int(self.config['model_settings']['secondary_fine_resolution'])
        self.core_pars['nlat1c'] = int(self.config['model_settings']['primary_coarse_resolution'])
        self.core_pars['nlat2c'] = int(self.config['model_settings']['secondary_coarse_resolution'])
        self.core_pars['npole'] = int(self.config['model_settings']['true_north_pole'])
        self.core_pars['roche1'] = int(self.config['model_settings']['primary_roche'])
        self.core_pars['roche2'] = int(self.config['model_settings']['secondary_roche'])
        self.core_pars['eclipse1'] = int(self.config['model_settings']['primary_eclipse'])
        self.core_pars['eclipse2'] = int(self.config['model_settings']['secondary_eclipse'])
        self.lcurve_model.set(self.core_pars)

    
    def vary_model_res(self, model_pars):
        model_pars['nlat1f'] = np.random.randint(self.core_pars['nlat1f'] - 5,
                                                 self.core_pars['nlat1f'] + 6)
        model_pars['nlat1c'] = np.random.randint(self.core_pars['nlat1c'] - 5,
                                                 self.core_pars['nlat1c'] + 6)
        model_pars['nlat2f'] = np.random.randint(self.core_pars['nlat2f'] - 5,
                                                 self.core_pars['nlat2f'] + 6)
        model_pars['nlat2c'] = np.random.randint(self.core_pars['nlat2c'] - 5,
                                                 self.core_pars['nlat2c'] + 6)


    def t2_free(self, band):
        if band == 'us' or band == 'u':
            return self.t2_u
        if band == 'gs' or band == 'g':
            return self.t2_g
        if band == 'rs' or band == 'r':
            return self.t2_r
        if band == 'is' or band == 'i':
            return self.t2_i
        if band == 'zs' or band == 'z':
            return self.t2_z

    def tcen_free(self, band, log_g1):
        if band == 'us' or band == 'u':
            return self.tcen_u
        if band == 'gs' or band == 'g':
            return self.tcen_g
        if band == 'rs' or band == 'r':
            return self.tcen_r
        if band == 'is' or band == 'i':
            return utils.get_Tbb(self.t1, log_g1, band, star_type='WD',
                                 source=self.config['wd_model'],
                                 instrument=self.config['filter_system'])
        if band == 'zs' or band == 'z':
            return utils.get_Tbb(self.t1, log_g1, band, star_type='WD',
                                 source=self.config['wd_model'],
                                 instrument=self.config['filter_system'])
    

    def slope(self, band):
        if band == 'us' or band == 'u':
            return 0.0
        if band == 'gs' or band == 'g':
            return self.slope_g
        if band == 'rs' or band == 'r':
            return self.slope_r
        if band == 'is' or band == 'i':
            return self.slope_i
        if band == 'zs' or band == 'z':
            return self.slope_z


    def get_value(self, band, factor=1.0):
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
        # lcurve_model = Lcurve(self.config['model_file'])
        lcurve_pars = dict()
        q = self.m2/self.m1

        # ENTER chosen mass-radius relations for both stars
        self.r1 = utils.get_radius(self.m1, self.t1, star_type=self.config['wd_core_comp'])
        self.r2 = utils.get_radius(self.m2, star_type='MS',
                                   relation=self.config['secondary_mr'], factor=factor)

        log_g1 = utils.log_g(self.m1, self.r1)
        log_g2 = utils.log_g(self.m2, self.r2)
        a = utils.separation(self.m1, self.m2, self.config['period'])

        if self.config['irr_infl'] == True:
            self.r2 = utils.irradiate(self.t1, self.r1, self.t2, self.r2, a)

        wd_model_flux = utils.integrate_disk(self.t1, log_g1, self.r1,
                                             self.parallax, self.ebv, band,
                                             self.config['wd_model'])
        lcurve_pars['t1'] = utils.get_Tbb(self.t1, log_g1, band, star_type='WD',
                                          source=self.config['wd_model'],
                                          instrument=self.config['filter_system'])
        lcurve_pars['t2'] = utils.get_Tbb(self.t2, log_g2, band, star_type='MS',
                                          instrument=self.config['filter_system'])
        # lcurve_pars['t2'] = self.t2_free(band)
        lcurve_pars['r1'] = self.r1/a  # scale to separation units
        lcurve_pars['r2'] = utils.Rva_to_Rl1(q, self.r2/a)  # scale and correct
        lcurve_pars['t0'] = self.t0
        lcurve_pars['period'] = self.config['period']
        lcurve_pars['tperiod'] = self.config['period']
        lcurve_pars['iangle'] = self.incl
        lcurve_pars['q'] = q
        # lcurve_pars['slope'] = self.slope(band)
        self.vary_model_res(lcurve_pars)
        if not self.config['fit_beta']:
            lcurve_pars['gravity_dark2'] = utils.get_gdc(self.t2, log_g2, band)
            # lcurve_pars['gravity_dark2'] = utils.get_gdc(self.t2_free(band), log_g2, band)
        else:
            lcurve_pars['gravity_dark2'] = utils.get_gdc(self.t2, log_g2, band, self.beta)
            # lcurve_pars['gravity_dark2'] = utils.get_gdc(self.t2_free(band), log_g2, band, self.beta)

        lcurve_pars['wavelength'] = self.cam.eff_wl[band].to_value(u.nm)
        lcurve_pars['phase1'] = (np.arcsin(lcurve_pars['r1']
                                           + lcurve_pars['r2'])
                                           / (2 * np.pi))+0.001
        lcurve_pars['phase2'] = 0.5 - lcurve_pars['phase1']

        self.lcurve_model.set(lcurve_pars)
        # self.lcurve_model.set(utils.get_ldcs(self.t1, logg_1=log_g1, band=band,
        #                                 star_type_1='WD', teff_2=self.t2_free(band),
        #                                 logg_2=log_g2, star_type_2='MS'))
        self.lcurve_model.set(utils.get_ldcs(self.t1, logg_1=log_g1, band=band,
                                        star_type_1='WD', teff_2=self.t2,
                                        logg_2=log_g2, star_type_2='MS'))

        # scale_factor = utils.scalefactor(a, self.parallax, wavelength=self.cam.eff_wl[band].to(u.nm), Ebv=self.ebv)
        if not self.lcurve_model.ok():
            raise ValueError('invalid parameter combination')
        ym, wdwarf = self.lcurve_model(self.lightcurves[band])

        # ym, wdwarf = self.lcurve_model(self.lightcurves[band], scale_factor)
        return ym, wdwarf, wd_model_flux

    def log_prior(self):
        """
        Prior probabilities
        """
        # first call parent class log_prior -> checks params in bounds
        val = super().log_prior()
        if np.isinf(val):
            return val

        defined_priors = [i for i, j in enumerate(self.parameter_names)
                          if j in list(self.config['priors'].keys())]
        for idx in defined_priors:
            var = list(self.parameter_names)[idx]
            prior = m.Prior(*self.config['priors'][var])
            val += prior.ln_prob(self.parameter_vector[idx])
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
        ym = self.get_value(band)[0]

        toff = int(np.floor(np.min(t)))
        tplot = t - toff

        if style == 'whole':
            # ax.errorbar(t, y, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.scatter(t, y, color=dcolor, marker='.', alpha=0.5)
            ax.plot(t, ym, color='k', lw=1, ls='-')
        elif style == 'residuals':
            # ax.errorbar(t, y-ym, yerr=ye, fmt='none', color=dcolor, alpha=0.5)
            ax.scatter(t, y-ym, color=dcolor, marker='.', alpha=0.5)
        else:
            raise ValueError('style not recognised')
    
    def plot_SED(self, ax, params):
        self.set_parameter_vector(params)
        wd_model = []
        wd_real = []
        wd_err = []
        wavelength = []
        for band in light_curves.keys():
            ym, wdwarf, wd_model_flux = self.get_value(band)
            ax.errorbar(self.cam.eff_wl[band].value, wdwarf,
                        yerr=self.flux_uncertainty[band]*wdwarf, c='k',
                        marker='.', elinewidth=1)
            ax.scatter(self.cam.eff_wl[band], wd_model_flux, c='r', marker='.')
            wd_model.append(wd_model_flux)
            wd_real.append(wdwarf)
            wd_err.append(self.flux_uncertainty[band]*wdwarf)
            wavelength.append(self.cam.eff_wl[band].value)
        out = np.column_stack((np.array(wavelength), np.array(wd_real), np.array(wd_err), np.array(wd_model)))
        # print(out)
        # np.savetxt('MCMC_runs/{}/{}_SED.dat'.format(self.config['run_name'],self.config['run_name']), out)


    
    def model(self, band, params):
        self.set_parameter_vector(params)
        t, _, y, ye, _, _ = np.loadtxt(self.lightcurves[band]).T
        ym = self.get_value(band)[0]
        return t, ym, y, ye

    
    def chisq(self, params, band, factor):
        """
        Calculate the chi-squared parameter.

        Parameters
        -----------
        params : iterable
            list of parameter values
        band : string
            SDSS/HiPERCAM band
        """
        self.set_parameter_vector(params)
        t, _, y, ye, w, _ = np.loadtxt(self.lightcurves[band]).T

        # check model params are valid - checks against bounds
        lp = self.log_prior()
        if not np.isfinite(lp):
            return np.inf
        else:
            try:
                ym, wdwarf, wd_flux = self.get_value(band, factor=factor)
                chisq = np.sum(((y - ym)/ye)**2)
                chisq += ((wdwarf - wd_flux)**2 / (wdwarf*self.flux_uncertainty[band])**2)
            except ValueError as err:
                # invalid lcurve params
                print('warning: model failed ', err)
                return np.inf
            # chisq = np.sum(w * ((y - ym)**2 / ye**2))
            return chisq


    def log_probability(self, params, band, factor):
        """
        Calculate log of posterior probability

        Parameters
        -----------
        params : iterable
            list of parameter values
        band : string
            SDSS/HiPERCAM band
        """
        # _, _, y, ye, w, _ = np.loadtxt(self.lightcurves[band]).T
        chisq = self.chisq(params, band, factor=factor)
        # log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * ye**2)) + chisq)
        log_likelihood = -0.5 * chisq
        return log_likelihood


    def write_model(self, params, band, fname):
        self.set_parameter_vector(params)

if __name__ == "__main__":
    import argparse
    from ruamel.yaml import YAML
    os.nice(5)
    
    # conf_file = 'GaiaDR2_4384149753578863744.yaml'
    conf_file = '2MASS_J1358-3556_irr_noparallax.yaml'
    # conf_file = 'ZTFJ2353_mag.yaml'
    # conf_file = 'CSS40190.yaml'
    # conf_file = 'SDSSJ1028_slope2.yaml'
    # conf_file = 'EC12250-3026.yaml'

    # parser.add_argument('--conf', '-c', action='store', default=conf_file)
    # args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open('config_files/{}'.format(conf_file)) as f:
    # with open('MCMC_runs/{}/{}.yaml'.format(conf_file, conf_file)) as f:
        config = yaml.load(f)
    run_settings = config['run_settings']

    parser = argparse.ArgumentParser(description='Fit or plot model of LC')
    parser.add_argument('--fit', '-f', action='store_true')
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--modelout', '-mo', action='store_true')
    parser.add_argument('--nwalkers', action='store', type=int, default=run_settings['walkers'])
    parser.add_argument('--nburn', action='store', type=int, default=run_settings['burnin'])
    parser.add_argument('--nprod', action='store', type=int, default=run_settings['production'])
    parser.add_argument('--nthreads', action='store', type=int, default=run_settings['n_cores'])
    args = parser.parse_args()


    run_name = config['run_name']
    chain_fname = run_name + '.chain'
    light_curves = config['light_curves']
    nameList = list(config['params'].keys())
    params = list(config['params'].values())
    ndim = len(params)

    model = EclipseLC(config, tuple(config['params'].keys()), *params, bounds=config['param_bounds'])

###############################################################################

    class clickevent:
        def __init__(self, list, fig):
            self.list = list
            self.cid = fig.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.inaxes is not None:
                self.list.append(event.xdata)
                if len(self.list) == 4:
                    plt.close()

    def model_chisq(params):
        chisq = 0
        model.set_parameter_vector(params)
        for band in light_curves.keys():
            chisq += model.chisq(params, band)
        return chisq

    def model_likelihood(params):
        val = 0
        model.set_parameter_vector(params)
        for band in light_curves.keys():
            val += model.log_probability(params, band)
        return val
    
    def model_prior(params):
        model.set_parameter_vector(params)
        return model.log_prior()

    # wrapper to combine log probability from all bands
    def log_probability(params):
        val = 0
        model.set_parameter_vector(params)
        if config['r2_pdf']:
            factor = np.random.normal(1.0, 0.05)
        else:
            factor = 1
        for band in light_curves.keys():
            val += model.log_probability(params, band, factor)
        val += model.log_prior()
        return val
    

    def plot_model(params, show=True, save=False, name='lightCurves.pdf'):
        gs = gridspec.GridSpec(len(list(light_curves.keys())), 2)
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
            color = sns.color_palette('nipy_spectral', 5)[iband-1]

            model.plot(ax_main, band, params, style='whole', dcolor=color)
            model.plot(ax_res, band, params, style='residuals', dcolor=color)
            if band != list(light_curves.keys())[-1]:
                plt.setp(ax_main.get_xticklabels(), visible=False)
                plt.setp(ax_res.get_xticklabels(), visible=False)
        if save:
            plt.savefig(name)
        if show:
            plt.show()
        plt.close()

    def plot(params, config, name='lightCurves_nice.pdf', dataname=None):

        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['font.size'] = 14

        band_colours = {
            'us': 'cornflowerblue',
            'u': 'cornflowerblue',
            'gs': 'limegreen',
            'g': 'limegreen',
            'rs': 'orange',
            'r': 'orange',
            'is': 'orangered',
            'i': 'orangered',
            'zs': 'darkred',
            'z': 'darkred'
        }

        annotate_str = {
            'us' : r'$u_{s}$',
            'gs' : r'$g_{s}$',
            'rs' : r'$r_{s}$',
            'is' : r'$i_{s}$',
            'zs' : r'$z_{s}$',
            'u' : r'$u$',
            'g' : r'$g$',
            'r' : r'$r$',
            'i' : r'$i$',
            'z' : r'$z$'
        }

        # data = load_data(conf_file, params)


        n_cols = len(light_curves.keys())
        fig = plt.figure(figsize=(n_cols * 3, 4))
        gs = gridspec.GridSpec(4, n_cols, figure=fig, wspace=0, hspace=0)
        gs.update(wspace=0, hspace=0)
        t0_idx = list(config['params'].keys()).index('t0')
        for idx in range(n_cols*2):
            idx_half = int(idx/2)
            band = list(light_curves.keys())[idx_half]
            t, ym, y, ye = model.model(band, params)
            if dataname:
                np.savetxt('{}_{}.dat'.format(dataname,band), np.column_stack((t,ym)))
            phase = t2phase(t, params[t0_idx], config['period'])

            if idx == 0:
                ax0 = fig.add_subplot(gs[:3, 0])
                ax = ax0
                max_abs_phase = np.max(np.abs(phase))
                ax0.set_xlim([-1.05*max_abs_phase, 1.05*max_abs_phase])
            elif idx == 1:
                ax1 = fig.add_subplot(gs[3:, 0], sharex=ax0)
                ax = ax1
            elif idx % 2 == 0 and idx != 0 and idx != 1:
                ax = fig.add_subplot(gs[:3, idx_half], sharey=ax0, sharex=ax0)
            else:
                ax = fig.add_subplot(gs[3:, idx_half], sharey=ax1, sharex=ax0)

            if idx % 2 == 0:
                ax.errorbar(phase, y*1e3, yerr=ye*1e3, lw=0, elinewidth=1,
                            marker='.', ms=3, zorder=1, color=band_colours[band])
                ax.plot(phase, ym*1e3, 'k-', lw=0.7, zorder=2)
                ax.axhline(0, c='k', ls='-', lw=0.3, zorder=2)
                ax.tick_params(top=False, bottom=True, left=True, right=True, direction='in')
                y_mid = (np.max(ax.get_ylim()) + np.min(ax.get_ylim())) / 2
                plt.setp(ax.get_xticklabels(), visible=False)

            else:
                ax.errorbar(phase, y*1e3 - ym*1e3, yerr=ye*1e3, lw=0,
                            elinewidth=1, marker='.', ms=2, zorder=1,
                            color=band_colours[band])
                ax.axhline(0, c='k', ls='--', zorder=2)
                ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
                yabs_max_res = np.max(np.abs(ax.get_ylim()))
                # ax.set_ylim(ymin=-yabs_max_res, ymax=yabs_max_res)
                ax.set_ylim(ymin=-0.035, ymax=0.035)
                # ax.set_xticks([-0.04, 0.00, 0.04])
            if idx_half != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        for axis, band in zip(fig.get_axes()[::2], list(light_curves.keys())):
            axis.annotate(annotate_str[band], xy=(0, y_mid), color='k', fontsize=18)

        plt.figtext(0.06, 0.5, 'Flux (mJy)', rotation='vertical')
        plt.figtext(0.47, 0.015, r'Orbital phase, $\phi$')
        plt.savefig(name, pad_inches=0.1, bbox_inches='tight')
        plt.close()

    def plot_SED(params, show=True, save=False, name='lightCurves.pdf'):

        ax = plt.subplot()
        model.plot_SED(ax, params)
        if save:
            plt.savefig(name)
        if show:
            plt.show()
        plt.close()


    if args.fit:
        
        folder = os.path.join('MCMC_runs', run_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        shutil.copyfile('config_files/{}'.format(conf_file),
                        'MCMC_runs/{}/{}.yaml'.format(run_name, run_name))
        chain_file = 'MCMC_runs/{}/{}'.format(run_name, chain_fname)
        nwalkers = args.nwalkers

        def log_prior(params):
            model.set_parameter_vector(params)
            return model.log_prior()

        # amount to scatter initial ball of walkers
        scatter = 0.001*np.ones_like(params)
        # small scatter for t0 and period
        scatter[nameList.index('t0')] = 1.0e-12
        # scatter[9] = 1.0e-9
        pool = Pool(args.nthreads)
        p0 = m.initialise_walkers(params, scatter, nwalkers, log_prior)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        state = None
        if args.nburn != 0:
            p0, prob, state = m.run_burnin(sampler, p0, args.nburn)
            sampler.reset()

        sampler = m.run_mcmc_save(sampler, p0, args.nprod, state, chain_file)
        fchain = m.flatchain(sampler.chain, ndim, thin=3)

        f = open('MCMC_runs/{}/{}.out'.format(run_name, run_name), 'w')
        lolim, medianPars, uplim = np.percentile(fchain, (16, 50, 84), axis=0)
        for name, par, lo, hi in zip(nameList, medianPars, lolim, uplim):
            print('{} = {} + {} - {}'.format(name, par, hi-par, par-lo))
            f.write('{} = {} + {} - {}\n'.format(name, par, hi-par, par-lo))
        f.close()

        fig = m.thumbPlot(fchain, nameList, hist_bin_factor=2)
        fig.savefig('MCMC_runs/{}/CP_{}.pdf'.format(run_name, run_name))
        plt.close()
        for i, name in enumerate(nameList):
            fig = m.plotchains(sampler.chain, i)
            fig.savefig('MCMC_runs/{}/Chain_{}_{}.pdf'.format(run_name, run_name, name))
            plt.close()
        # import plot_lcs
        plot(medianPars[:-1], config, 'MCMC_runs/{}/LC_{}.pdf'.format(run_name, run_name),
             dataname='MCMC_runs/{}/model_{}'.format(run_name, run_name))
        plot_SED(medianPars[:-1], show=False, save=True,
                 name='MCMC_runs/{}/SED_{}.pdf'.format(run_name, run_name))
        plot_model(medianPars[:-1], show=False, save=False,
                   name='MCMC_runs/{}/LC_{}.pdf'.format(run_name, run_name))


    elif args.test:
        print('Model has ln_prob of {}'.format(log_probability(params)))
        plot_SED(params, show=True, save=False)
        plot_model(params, show=True, save=False, name='MCMC_runs/test_lc.pdf')


    elif args.modelout:
        for iband, band in enumerate(light_curves.keys()):
            t, _, y, ye, _, _ = np.loadtxt(model.lightcurves[band]).T
            ym, _, _, _ = model.get_value(band)
            model_out = np.vstack((t, ym, y, ye)).T
            np.savetxt('{}_model_lc_{}.dat'.format(run_name, band), model_out)

    else:
        chain_file = 'MCMC_runs/{}/{}'.format(run_name, chain_fname)
        chain = m.readchain(chain_file)[10000:, :, :]
        fchain = m.flatchain(chain, ndim+1)
        print(chain.shape)
        print(fchain.shape)
        namelist = nameList.append('ln_prob')
        medianPars = np.median(fchain, axis=0)

        f = open('MCMC_runs/{}/{}.out'.format(run_name, run_name), 'w')
        lolim, medianPars, uplim = np.percentile(fchain, (16, 50, 84), axis=0)
        for name, par, lo, hi in zip(nameList, medianPars, lolim, uplim):
            print('{} = {} + {} - {}'.format(name, par, hi-par, par-lo))
            f.write('{} = {} + {} - {}\n'.format(name, par, hi-par, par-lo))
        f.close()

        fig = m.thumbPlot(fchain, nameList, hist_bin_factor=2)
        fig.savefig('MCMC_runs/{}/CP_{}.pdf'.format(run_name, run_name))
        plt.close()
        for i, name in enumerate(nameList):
            fig = m.plotchains(chain, i)
            fig.savefig('MCMC_runs/{}/Chain_{}_{}.pdf'.format(run_name, run_name, name))
            plt.close()
        # import plot_lcs
        plot(medianPars[:-1], config, 'MCMC_runs/{}/LC_{}.pdf'.format(run_name, run_name),
             dataname='MCMC_runs/{}/model_{}'.format(run_name, run_name))
        plot_SED(medianPars[:-1], show=False, save=True,
                 name='MCMC_runs/{}/SED_{}.pdf'.format(run_name, run_name))
        plot_model(medianPars[:-1], show=False, save=False,
                   name='MCMC_runs/{}/LC_{}.pdf'.format(run_name, run_name))