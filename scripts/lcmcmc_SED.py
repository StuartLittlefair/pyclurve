import emcee
from multiprocessing import Pool
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import shutil
from astropy.table import Table

from pylcurve.lcurve import Lcurve
from pylcurve.modelling import Model
import pylcurve.mcmc_utils as m
import pylcurve.utils as utils
from pylcurve.filters import filters
import pylcurve.plotting as p


"""
This script fits flux calibrated, multi-band primary eclipse photometry of
WD-WD/WD-dM binaries using an MCMC method to run Tom Marsh's LROCHE routine.
Using mass-radius relations it can determine stellar masses and effective
temperatures for both components.
"""


class EclipseLC(Model):

    def __init__(self, config, *args, **kwargs):
        """from pylcurve.filters import filtersms
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
        self.config = config
        self.lightcurves = self.config['light_curves']
        self.cam = filters(self.config['filter_system'])
        self.flux_uncertainty = self.config['flux_uncertainty']
        self.lcurve_model = Lcurve(self.config['model_file'])
        self.set_core_parameters()
        super().__init__(*args, **kwargs)
    

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

        if not self.lcurve_model.ok():
            raise ValueError('invalid parameter combination')
        ym, wdwarf = self.lcurve_model(self.lightcurves[band])

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
    
    conf_file = '1712af_CO_corr.yaml'


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


    def write_print_output(run_name, fchain, namelist):
        f = open(f"MCMC_runs/{run_name}/{run_name}.out", 'w')
        lolim, medianPars, uplim = np.percentile(fchain, (16, 50, 84), axis=0)
        for name, par, lo, hi in zip(namelist, medianPars, lolim, uplim):
            print(f"{name} = {par} + {hi-par} - {par-lo}")
            f.write(f"{name} = {par} + {hi-par} - {par-lo}\n")
        f.close()


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

        write_print_output(run_name, fchain, nameList)
        medianPars = np.median(fchain, axis=0)

        # make plots
        p.plot_CP(fchain, nameList, name=f"MCMC_runs/{run_name}/CP_{run_name}.pdf")
        p.plot_traces(sampler.chain, nameList, name=f"MCMC_runs/{run_name}/Trace_{run_name}.pdf")
        p.plot_LC(model, medianPars[:-1], f"MCMC_runs/{run_name}/LC_{run_name}.pdf",
                  dataname=f"MCMC_runs/{run_name}/model_{run_name}")
        p.plot_SED(model, medianPars[:-1], show=False, save=True,
                   name=f"MCMC_runs/{run_name}/SED_{run_name}.pdf")


    elif args.test:
        print('Model has ln_prob of {}'.format(log_probability(params)))
        p.plot_SED(model, params, show=True, save=False)
        p.plot_LC(model, params, show=True, save=False)


    elif args.modelout:
        for band in enumerate(light_curves.keys()):
            t, ym, y, ye = model.model(band, params)
            model_out = np.vstack((t, ym, y, ye)).T
            np.savetxt('{}_model_lc_{}.dat'.format(run_name, band), model_out)

    else:
        chain_file = 'MCMC_runs/{}/{}'.format(run_name, chain_fname)
        chain = m.readchain(chain_file)[1000:, :, :]
        fchain = m.flatchain(chain, ndim+1)
        print(chain.shape)
        nameList.append('ln_prob')
        medianPars = np.median(fchain, axis=0)

        write_print_output(run_name, fchain, nameList) # print & save output

        # make plots
        p.plot_traces(chain, nameList, name=f"MCMC_runs/{run_name}/Trace_{run_name}.pdf")
        p.plot_CP(fchain, nameList, name=f"MCMC_runs/{run_name}/CP_{run_name}.pdf")
        p.plot_LC(model, medianPars[:-1], f"MCMC_runs/{run_name}/LC_{run_name}.pdf",
                dataname=f"MCMC_runs/{run_name}/model_{run_name}")
        p.plot_SED(model, medianPars[:-1], show=False, save=True,
                 name=f"MCMC_runs/{run_name}/SED_{run_name}.pdf")