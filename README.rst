LCURVE Python 'wrapper'
===================================

The tools here provide a thin Python wrapper around Tom Marsh's LCURVE. The wrapper really
does nothing more complex than provide a Python class which reads in an ```lcurve``` model file
and allows you to call Tom's ```lroche``` to evaluate the model.

There's also some utility classes to help you write your own models that ultimately call ```lroche```,
and some helper functions to aid in running MCMC fits. You can look at ```lcmcmc.py``` in the scripts
directory for an example that uses all this functionality to simultaneously fit the 5 HiPERCAM bands
simultaneously for a double WD eclipsing binary, with a Gaussian Process representation of the
pulsations seen in one component.

Installation
------------

The software is written as much as possible to make use of core Python
components. The third-party requirements are:

- `astropy <http://astropy.org/>`_, a package for astronomical calculations;

- `emcee <http://http://dfm.io/emcee/current/>`_ for MCMC fitting;

- `celerite <https://celerite.readthedocs.io/en/stable/>`_, used for scalable Gaussian processes.


Usually, installing with pip will handle these dependencies for you, so installation is a simple matter of typing::

 pip install pylcurve

or if you don't have root access::

 pip install --prefix=my_own_installation_directory pylcurve

Finally, since this is a wrapper around Tom's LCURVE package, you must tell the wrapper where to find
LCURVE! Set the `TRM_SOFTWARE` environment variable to point at the root directory of the LCURVE installation.
For example, if lcurve is installed in `/home/user/trm_soft/bin/lcurve` then::

 setenv TRM_SOFTWARE /home/user/trm_soft
