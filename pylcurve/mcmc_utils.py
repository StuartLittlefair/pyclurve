import numpy as np
import scipy.stats as stats
import emcee
import corner as triangle
import pandas as pd
# lightweight progress bar
from tqdm import tqdm
import scipy.integrate as intg
import warnings
from matplotlib import pyplot as plt


TINY = -np.inf


class Prior(object):
    '''a class to represent a prior on a parameter, which makes calculating
    prior log-probability easier.

    Priors can be of five types: gauss, gaussPos, uniform, log_uniform and mod_jeff

    gauss is a Gaussian distribution, and is useful for parameters with
    existing constraints in the literature
    gaussPos is like gauss but enforces positivity
    Gaussian priors are initialised as Prior('gauss',mean,stdDev)

    uniform is a uniform prior, initialised like Prior('uniform',low_limit,high_limit)
    uniform priors are useful because they are 'uninformative'

    log_uniform priors have constant probability in log-space. They are the uninformative prior
    for 'scale-factors', such as error bars (look up Jeffreys prior for more info)

    mod_jeff is a modified jeffries prior - see Gregory et al 2007
    they are useful when you have a large uncertainty in the parameter value, so
    a jeffreys prior is appropriate, but the range of allowed values starts at 0

    they have two parameters, p0 and pmax.
    they act as a jeffrey's prior about p0, and uniform below p0. typically
    set p0=noise level
    '''
    def __init__(self, type, p1, p2):
        assert type in ['gauss', 'gaussPos', 'uniform', 'log_uniform', 'mod_jeff', 'log_normal']
        self.type = type
        self.p1 = p1
        self.p2 = p2
        if type == 'log_uniform' and self.p1 < 1.0e-30:
            warnings.warn('lower limit on log_uniform prior rescaled from %f to 1.0e-30' % self.p1)
            self.p1 = 1.0e-30
        if type == 'log_uniform':
            self.normalise = 1.0
            self.normalise = np.fabs(intg.quad(self.ln_prob, self.p1, self.p2)[0])
        if type == 'mod_jeff':
            self.normalise = np.log((self.p1+self.p2)/self.p1)

    def ln_prob(self, val):
        if self.type == 'gauss':
            p = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
            if p > 0:
                return np.log(stats.norm(scale=self.p2, loc=self.p1).pdf(val))
            else:
                return TINY
        elif self.type == 'log_normal':
            if val < 1.0e-30:
                warnings.warn('evaluating log_normal prior on val %f. Rescaling to 1.0e-30' % val)
                val = 1.0e-30
            log_val = np.log10(val)
            p = stats.norm(scale=self.p2, loc=self.p1).pdf(log_val)
            if p > 0:
                return np.log(stats.norm(scale=self.p2, loc=self.p1).pdf(log_val))
            else:
                return TINY
        elif self.type == 'gaussPos':
            if val <= 0.0:
                return TINY
            else:
                p = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
                if p > 0:
                    return np.log(p)
                else:
                    return TINY
        elif self.type == 'uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0/np.abs(self.p1-self.p2))
            else:
                return TINY
        elif self.type == 'log_uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0 / self.normalise / val)
            else:
                return TINY
        elif self.type == 'mod_jeff':
            if (val > 0) and (val < self.p2):
                return np.log(1.0 / self.normalise / (val+self.p1))
            else:
                return TINY


def fracWithin(pdf, val):
    return pdf[pdf >= val].sum()


def thumbPlot(chain, labels, **kwargs):
    fig = triangle.corner(chain, labels=labels, **kwargs)
    return fig


def initialise_walkers(p, scatter, nwalkers, ln_prior):
    """
    Create starting ball of walkers with a certain amount of scatter

    Ball of walkers respects the prior, so all starting positions are valid

    Parameters
    ----------
    p : list or np.ndarray
        starting parameters
    scatter : float or np.ndarray
        amplitude of random scatter. Use an array if you want different amounts
        of scatter for different parameters
    nwalkers : int
        number of emcee walkers (i.e semi-independent MCMC chains)
    ln_prior : callable
        A function to evaluate prior probability for each parameter. Accepts a
        single argument which must be the same form as p and returns a float, or
        -np.inf if the parameter combination violates the priors.

    Returns
    -------
    p0 : np.ndarray
        Starting ball for MCMC. Shape is (nwalkers, npars).
    """
    p0 = emcee.utils.sample_ball(p, scatter*p, size=nwalkers)
    # Make initial number of invalid walkers equal to total number of walkers
    numInvalid = nwalkers
    print('Initialising walkers...')
    print('Number of walkers currently invalid:')
    # All invalid params need to be resampled
    while numInvalid > 0:
        # Create a mask of invalid params
        isValid = np.array([np.isfinite(ln_prior(p)) for p in p0])
        bad = p0[~isValid]
        # Determine the number of good and bad walkers
        nbad = len(bad)
        print(nbad)
        ngood = len(p0[isValid])
        # Choose nbad random rows from ngood walker sample
        replacement_rows = np.random.randint(ngood, size=nbad)
        # Create replacement values from valid walkers
        replacements = p0[isValid][replacement_rows]
        # Add scatter to replacement values
        replacements += 0.5*replacements*scatter*np.random.normal(size=replacements.shape)
        # Replace invalid walkers with new values
        p0[~isValid] = replacements
        numInvalid = len(p0[~isValid])
    return p0


def run_burnin(sampler, startPos, nSteps, storechain=False, progress=True):
    """
    Runs burn-in phase of MCMC, with options to store the chain or show progress bar
    """
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, state in sampler.sample(startPos, iterations=nSteps, storechain=storechain):
        iStep += 1
        if progress:
            bar.update()
    return pos, prob, state


def run_mcmc_save(sampler, startPos, nSteps, rState, file, progress=True, **kwargs):
    """
    Runs an MCMC chain with emcee, and saves steps to a file
    """
    # open chain save file
    if file:
        f = open(file, "w")
        f.close()
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, state in sampler.sample(startPos, iterations=nSteps,
                                           rstate0=rState, storechain=True, **kwargs):
        if file:
            f = open(file, "a")
        iStep += 1
        if progress:
            bar.update()
        for k in range(pos.shape[0]):
            # loop over all walkers and append to file
            thisPos = pos[k]
            thisProb = prob[k]
            if file:
                f.write("{0:4d} {1:s} {2:f}\n".format(k, " ".join(map(str, thisPos)), thisProb))
        if file:
            f.close()
    return sampler


def flatchain(chain, npars, nskip=0, thin=1):
    '''
    Flattens a chain (i.e collects results from all walkers)

    Options exist to skip the first nskip parameters, and thin the chain
    by only retrieving a point every thin steps - thinning can be useful when
    the steps of the chain are highly correlated
    '''
    return chain[:, nskip::thin, :].reshape((-1, npars))


def readchain(file, nskip=0, thin=1):
    data = pd.read_csv(file, header=None, compression=None, delim_whitespace=True)
    data = np.array(data)
    nwalkers = int(data[:, 0].max()+1)
    nprod = int(data.shape[0]/nwalkers)
    npars = data.shape[1] - 1  # first is walker ID, last is ln_prob
    chain = np.reshape(data[:, 1:], (nwalkers, nprod, npars))
    return chain


def readflatchain(file):
    data = pd.read_csv(file, header=None, compression=None, delim_whitespace=True)
    data = np.array(data)
    return data


def plotchains(chain, npar, alpha=0.2):
    nwalkers, nsteps, npars = chain.shape
    fig = plt.figure()
    for i in range(nwalkers):
        plt.plot(chain[i, :, npar], alpha=alpha, color='k')
    return fig


def GR_diagnostic(sampler_chain):
    '''Gelman & Rubin check for convergence.'''
    m, n, ndim = np.shape(sampler_chain)
    R_hats = np.zeros((ndim))
    samples = sampler_chain[:, :, :].reshape(-1, ndim)
    for i in range(ndim):  # iterate over parameters

        # Define variables
        chains = sampler_chain[:, :, i]

        flat_chain = samples[:, i]
        psi_dot_dot = np.mean(flat_chain)
        psi_j_dot = np.mean(chains, axis=1)
        psi_j_t = chains

        # Calculate between-chain variance
        between = sum((psi_j_dot - psi_dot_dot)**2) / (m - 1)

        # Calculate within-chain variance
        inner_sum = np.sum(np.array([(psi_j_t[j, :] - psi_j_dot[j])**2 for j in range(m)]), axis=1)
        outer_sum = np.sum(inner_sum)
        W = outer_sum / (m*(n-1))

        # Calculate sigma
        sigma2 = (n-1)/n * W + between

        # Calculate convergence criterion (potential scale reduction factor)
        R_hats[i] = (m + 1)*sigma2/(m*W) - (n-1)/(m*n)
    return R_hats
