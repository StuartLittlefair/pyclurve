import emcee
from multiprocessing import Pool, cpu_count
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
        """
        Sets lcurve's computational parameters from those specified
        in the config file.
        """
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
        """
        Varies the resolution of lcurve's stellar grids to try and blur out any
        systematic grid effects that could cause issues for the inclination
        and radii.
        """
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


    def set_model(self, band, factor=1.0):
        """"
        Set model from given parameters

        Parameters
        ----------
        band : string
            SDSS/HiPERCAM bandpass
        
        factor : float
            Factor to scale radius of secondary star

        """
        lcurve_pars = dict()
        q = self.m2/self.m1

        # ENTER chosen mass-radius relations for both stars
        self.r1 = utils.get_radius(self.m1, self.t1, star_type=self.config['wd_core_comp'])
        self.r2 = utils.get_radius(self.m2, star_type='MS',
                                   relation=self.config['secondary_mr'], factor=factor)

        self.log_g1 = utils.log_g(self.m1, self.r1)
        self.log_g2 = utils.log_g(self.m2, self.r2)
        self.a = utils.separation(self.m1, self.m2, self.config['period'])

        if self.config['free_t2'] == True:
            self.t2 = self.t2_free(band)
            t2 = self.t2_g
        else:
            t2 = self.t2

        if self.config['irr_infl'] == True:
            self.r2 = utils.irradiate(self.t1, self.r1, t2, self.r2, self.a)

        lcurve_pars['t1'] = utils.get_Tbb(self.t1, self.log_g1, band, star_type='WD',
                                          source=self.config['wd_model'],
                                          instrument=self.config['filter_system'])
        if self.config['free_t2'] == False:
            lcurve_pars['t2'] = utils.get_Tbb(self.t2, self.log_g2, band, star_type='MS',
                                            instrument=self.config['filter_system'])
        else:
            lcurve_pars['t2'] = self.t2

        lcurve_pars['r1'] = self.r1/self.a  # scale to separation units
        lcurve_pars['r2'] = utils.Rva_to_Rl1(q, self.r2/self.a)  # scale and correct
        lcurve_pars['t0'] = self.t0
        lcurve_pars['period'] = self.config['period']
        lcurve_pars['tperiod'] = self.config['period']
        lcurve_pars['iangle'] = self.incl
        lcurve_pars['q'] = q
        # lcurve_pars['slope'] = self.slope(band)
        self.vary_model_res(lcurve_pars)
        if not self.config['fit_beta']:
            lcurve_pars['gravity_dark2'] = utils.get_gdc(t2, self.log_g2, band)
        else:
            lcurve_pars['gravity_dark2'] = utils.get_gdc(t2, self.log_g2, band, self.beta)

        lcurve_pars['wavelength'] = self.cam.eff_wl[band].to_value(u.nm)
        lcurve_pars['phase1'] = (np.arcsin(lcurve_pars['r1']
                                           + lcurve_pars['r2'])
                                           / (2 * np.pi))+0.001
        lcurve_pars['phase2'] = 0.5 - lcurve_pars['phase1']

        self.lcurve_model.set(lcurve_pars)
        # self.lcurve_model.set(utils.get_ldcs(self.t1, logg_1=self.log_g1, band=band,
        #                                 star_type_1='WD', teff_2=self.t2_free(band),
        #                                 logg_2=self.log_g2, star_type_2='MS'))
        self.lcurve_model.set(utils.get_ldcs(self.t1, logg_1=self.log_g1, band=band,
                                             star_type_1='WD', teff_2=t2,
                                             logg_2=self.log_g2, star_type_2='MS'))

        if not self.lcurve_model.ok():
            raise ValueError('invalid parameter combination')
        

    def get_value(self, band, factor=1.0):
        """
        Calculate lightcurve

        Parameters
        ----------
        band : string
            SDSS/HiPERCAM bandpass
        
        factor : float
            Factor to scale radius of secondary star

        Returns
        -------
        ym : np.ndarray
            Model light curve flux

        wdwarf : float
            WD contribution to light curve

        wd_model_flux : float
            Theoretical WD flux for model parameters

        """
        self.set_model(band, factor)
        wd_model_flux = utils.integrate_disk(self.t1, self.log_g1, self.r1,
                                             self.parallax, self.ebv, band,
                                             self.config['wd_model'])
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
        factor : float
            Factor to scale radius of secondary star by
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
                chisq = np.sum(w * ((y - ym)/ye)**2)
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
        factor : float
            Factor to scale radius of secondary star by
        """
        # _, _, y, ye, w, _ = np.loadtxt(self.lightcurves[band]).T
        chisq = self.chisq(params, band, factor=factor)
        # log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * ye**2)) + chisq)
        log_likelihood = -0.5 * chisq
        return log_likelihood


    def write_model(self, band, fname):
        self.set_model(band)
        self.lcurve_model.write(fname)


if __name__ == "__main__":
    import argparse
    from ruamel.yaml import YAML
    os.nice(5)

    folders = ['config_files', 'light_curves', 'MCMC_runs', 'model_files']    
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    conf_file = 'example.yaml'


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

    # make sure some CPUs are left free.
    n_cpu = cpu_count()
    if args.nthreads >= n_cpu - 1:
        args.nthreads = n_cpu - 2

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

    
    def inf_to_small(x):
        if x == -np.inf:
            return -1e20
        else:
            return x


    def min_func(x):
        return -inf_to_small(log_probability(x))


    def fit_start_pos(x0, *args, **kwargs):
        bounds = [tuple(value) for value in config['param_bounds'].values()]
        print("Fitting start position...")
        soln = minimize(min_func, np.array(x0), method='Nelder-Mead', bounds=bounds, *args, **kwargs)
        return soln.x

    
    def clip_lnprob(chain, clip_margin=100):
        med = np.median(chain, axis=(0,1))
        idx = np.argwhere(np.min(chain[:,:,-1], axis=0) < med[-1] - clip_margin)
        chain = np.delete(chain, idx.flatten(), axis=1)
        return chain

    
    def write_models(params, fname):
        model.set_parameter_vector(params)
        for band in light_curves.keys():
            fname_out = f"{fname}_{band}.mod"
            model.write_model(band, fname_out)


    def write_print_output(run_name, fchain, namelist):
        f = open(f"MCMC_runs/{run_name}/{run_name}.out", 'w')
        lolim, medianPars, uplim = np.percentile(fchain, (16, 50, 84), axis=0)
        for name, par, lo, hi in zip(namelist, medianPars, lolim, uplim):
            print(f"{name} = {par} + {hi-par} - {par-lo}")
            f.write(f"{name} = {par} + {hi-par} - {par-lo}\n")
        f.close()

    
    def mcmc_results(chain_file, par_names, run_name, burn_in=1000, thin=1, measure='median'):
        chain = m.readchain(chain_file)[burn_in:, :, :]
        par_names.append('ln_prob')
        ndim = chain.shape[-1]
        fchain = m.flatchain(chain, ndim, thin=thin)

        if measure == 'median':
            Pars = np.median(fchain, axis=0)
        elif measure == 'best':
            Pars = fchain[np.argmax(fchain[:,-1]), :]
        
        write_print_output(run_name, fchain, par_names)

        # make plots
        p.plot_traces(chain, par_names, name=f"MCMC_runs/{run_name}/Trace_{run_name}.pdf")
        p.plot_CP(fchain, par_names, name=f"MCMC_runs/{run_name}/CP_{run_name}.pdf")
        p.plot_LC(model, Pars[:-1], f"MCMC_runs/{run_name}/LC_{run_name}.pdf",
                  dataname=f"MCMC_runs/{run_name}/model_{run_name}")
        p.plot_SED(model, Pars[:-1], show=False, save=True,
                   name=f"MCMC_runs/{run_name}/SED_{run_name}.pdf")
        write_models(Pars[:-1], fname=f"MCMC_runs/{run_name}/{run_name}")


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

        # fit for start position close to best log_prob
        params = fit_start_pos(np.array(params), tol=1)

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

        mcmc_results(chain_file, nameList, run_name, burn_in=0, measure='median') # sampler.get_chain()

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
        mcmc_results(chain_file, nameList, run_name, burn_in=0, measure='median')