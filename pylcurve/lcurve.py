from collections import OrderedDict
import tempfile
import os
import subprocess
from trm import roche
import numpy as np

trm_sw = os.environ.get('TRM_SOFTWARE', '/usr/local/ultracam')
lroche = os.path.join(trm_sw, 'bin', 'lcurve', 'lroche')
tcontrast = os.path.join(trm_sw, 'bin', 'lcurve', 'tcontrast')


class Lcurve(OrderedDict):
    """
    Class representing an Lcurve model. Initialised by reading
    in an Lcurve model file.
    """

    def __init__(self, model, prior=None):
        """
        model  -- the name of the model file to initialise this
        prior  -- a function to return -2*log(prior prob) to add to the
                  chi**2 during trials. This is a function that will be passed
                  a dictionary of parameter name/value pairs. Here is an
                  example applying a prior constraint on a parameter 'q':
                  def prior(pdict):
                     return ((pdict['q']-0.1)/0.05)**2
                  which imposes a gaussian prior mean 0.1, RMS=0.05
                  Note that the prior can be relative to its peak value so no
                  normalisation terms are needed.
        args   -- a tuple of extra arguments to pass through to 'prior' if
                  necessary. These come after name/value dictionary argument.
        """
        super().__init__(self)
        fptr = open(model)
        for line in fptr:
            eq = line.find('=')
            if eq > -1:
                self[line[:eq].strip()] = line[eq+1:].strip().split()
        fptr.close()
        self.model = model
        self.prior = prior


    def __reduce__(self):
        """
        OrderedDict.__reduce__ overrides the normal reduce.
        Necessary to implement this to allow pickling.
        """
        state = super().__reduce__()
        # OrderedDict.__reduce__ returns a 5 tuple
        # the first and last can be kept
        # the fourth is None and needs to stay None
        # the second must be set to a sequence of arguments to __init__
        # the third can be used to store additional attributes
        newstate = (state[0],
                    (self.model, self.prior),
                    None,
                    None,
                    state[4])
        return newstate


    def set(self, p):
        """
        Sets the model to a dictionary of parameter values.
        """
        for key, value in p.items():
            if key in self:
                self[key][0] = value
            else:
                raise Exception('Lcurve.set: could not find parameter = '
                                + key + ' in current model.')


    def get(self, key):
        return float(self[key][0])


    def vcheck(self, name, vmin, vmax):
        """Checks that a parameter of a given name is in range."""
        v = float(self[name][0])
        return (v >= vmin and v <= vmax)


    def ok(self):
        """
        Checks for silly parameter values. Runnning this before chisq could
        save time and disaster.
        """
        return (self.vcheck('q', 0.001, 100.) and self.vcheck('iangle', 0., 90.)
            and self.vcheck('r1', 0., 1.)
            and (float(self['r2'][0]) <= 0.
                 or self.vcheck('r2', 0., 1-roche.xl1(float(self['q'][0]))))
            and self.vcheck('cphi3', 0., 0.25)
            and self.vcheck('cphi4', float(self['cphi3'][0]), 0.25)
            and self.vcheck('t1', 0., 1.e6)
            and self.vcheck('t2', -1.e6, 1.e6)
            and self.vcheck('ldc1_1', -50., 50.)
            and self.vcheck('ldc1_2', -50., 50.)
            and self.vcheck('ldc1_3', -50., 50.)
            and self.vcheck('ldc1_4', -50., 50.)
            and self.vcheck('ldc2_1', -50., 50.)
            and self.vcheck('ldc2_2', -50., 50.)
            and self.vcheck('ldc2_3', -50., 50.)
            and self.vcheck('ldc2_3', -50., 50.)
            and self.vcheck('period', 0., 100.)
            and self.vcheck('gravity_dark1', 0., 1.)
            and self.vcheck('gravity_dark2', 0., 1.)
            and (
                 self['add_disc'][0] == '0'
                 or (
                     (float(self['rdisc1'][0]) <= 0.
                      or self.vcheck('rdisc1', float(self['r1'][0]), 1.0))
                      and (float(self['rdisc2'][0]) <= 0.
                      or self.vcheck('rdisc2', float(self['rdisc1'][0]), 1.))
                      and self.vcheck('temp_disc', 0., 1.e6)
                      and self.vcheck('texp_disc', -10., 10.)
                    )
                )
            and (
                 self['add_spot'][0] == '0'
                 or (self.vcheck('radius_spot', 0., 0.85*roche.xl1(float(self['q'][0])))
                     and self.vcheck('length_spot', 0., 10.)
                     and self.vcheck('expon_spot', 0., 30.)
                     and self.vcheck('temp_spot', 0., 1.e6)
                     and self.vcheck('cfrac_spot', 0., 1.)
                     and self.vcheck('epow_spot', 0., 10.)
                    )
                ))


    def write(self, fname=None):
        """
        Writes model to a file, or a temporary file if fname=None.
        It returns the name of the file.
        You should delete the temporary file at some point.
        """
        if not fname:
            (fd, fname) = tempfile.mkstemp()
            fptr = os.fdopen(fd, 'w')
        else:
            fptr = open(fname, 'w')

        for key, value in self.items():
            fptr.write('%-20s =' % key)
            for item in value:
                fptr.write(' ' + str(item))
            fptr.write('\n')
        fptr.close()
        return fname


    def run(self, data, scale_factor=None):
        """
        Run lroche, return output and name of model.
        """
        data = str(data)
        if not os.path.isfile(data):
            raise Exception(data + ' does not appear to be a file')

        # write the model to a temporary file
        tfile = self.write()

        (fd, fname) = tempfile.mkstemp()

        # build the command, run it, read the results.
        if scale_factor:
            args = (lroche, 'model=' + tfile, 'data=' + data, 'noise=0',
                    'scale=no', 'ssfac={}', 'seed=12345', 'nfile=1',
                    'output={}'.format(scale_factor, fname), 'device=null',
                    'nodefs')
        else:
            args = (lroche, 'model=' + tfile, 'data=' + data, 'noise=0',
                    'scale=yes', 'seed=12345', 'nfile=1',
                    'output={}'.format(fname), 'device=null', 'nodefs')

        p = subprocess.Popen(args, close_fds=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        sout, serr = p.communicate()
        output = sout.decode().split('\n')

        # delete the temporary file
        os.remove(tfile)
        os.close(fd)

        return output, fname


    def tcontrast(self):
        """
        Run tcontrast, and get mean dayside and nightside temperatures.

        Returns
        -------
        tmax, tday, tnight, tmin: temperatures from LROCHE surface map
        of companion.
        """
        # write the model to a temporary file
        tfile = self.write()

        args = (tcontrast, 'model=' + tfile)
        p = subprocess.Popen(args, close_fds=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        sout, serr = p.communicate()
        output = sout.decode().split('\n')

        # delete the temporary file
        os.remove(tfile)
        return [float(val) for val in output[0].split()]


    def residuals(self, data, scale_factor=None):
        output, fname = self.run(data, scale_factor)
        x, dx, y, e, _, _ = np.loadtxt(data).T
        _, _, ym, _, _, _ = np.loadtxt(fname).T
        os.remove(fname)
        return x, y-ym, e


    def __call__(self, data, scale_factor=None):
        output, fname = self.run(data, scale_factor)
        t, et, y, ye, _, _ = np.loadtxt(data).T
        _, _, ym, _, _, _ = np.loadtxt(fname).T
        os.remove(fname)
        return t, y, ye, ym


    def chisq(self, data, scale_factor=None):
        """
        Computes chi**2 of the current model against a given
        data file. This first writes out the current model to
        a temporary disc file which it then passes to a call
        to lroche along with the name of the data file. It returns
        (chisq, wnok, wdwarf) where chisq and wnok are the weighted
        chi**2 and number of data points while wdwarf
        is the white dwarf's contribution as reported by lroche.
        If something goes wrong, it comes back with 'None'
        in each of these values.
        """
        output, fname = self.run(data, scale_factor=None)
        os.remove(fname)

        # interpret the output
        subout = [out for out in output if out.startswith('Weighted')]

        if len(subout) == 1:
            eq = subout[0].find('=')
            comma = subout[0].find(',')
            chisq = float(subout[0][eq+2:comma])
            wnok = float(subout[0][subout[0].find('wnok =')+6:].strip())
            subout = [out for out in output if out.startswith('White dwarf')]
            if len(subout) == 1:
                eq = subout[0].find('=')
                wdwarf = float(subout[0][eq+2:])
            else:
                print("Can't find white dwarf's contribution. lroche out of date.")
                chisq = wnok = wdwarf = None

        else:
            chisq = wnok = wdwarf = None
            print('Output from lroche failure')

        return chisq, wnok, wdwarf


    def var(self):
        """
        Returns dictionary of current values of variable parameters
        keyed on the parameter name.
        """
        vpar = OrderedDict()
        use_radii = (self['use_radii'][0] == '1')
        for (key, value) in self.items():
            if len(value) > 1 and value[3] == '1':
                if (use_radii and (key == 'r1' or key == 'r2'))
                   or (not use_radii and (key == 'cphi3' or key == 'cphi4')) \
                   or (key != 'r1' and key != 'r2'
                       and key != 'cphi3' and key != 'cphi4'):
                    vpar[key] = float(value[0])

        return vpar


    def vvar(self):
        """
        Returns names of variable parameters
        """
        vnam = []
        use_radii = (self['use_radii'][0] == '1')
        for (key, value) in self.items():
            if len(value) > 1 and value[3] == '1':
                if (use_radii and (key == 'r1' or key == 'r2'))
                   or (not use_radii and (key == 'cphi3' or key == 'cphi4')) \
                   or (key != 'r1' and key != 'r2'
                       and key != 'cphi3' and key != 'cphi4'):
                    vnam.append(key)

        return vnam


    def nvar(self):
        """
        Returns the number of variable parameters
        """
        nvar = 0
        use_radii = (self['use_radii'][0] == '1')
        for (key, value) in self.items():
            if len(value) > 1 and value[3] == '1':
                if (use_radii and (key == 'r1' or key == 'r2'))
                   or (not use_radii and (key == 'cphi3' or key == 'cphi4')) \
                   or (key != 'r1' and key != 'r2'
                       and key != 'cphi3' and key != 'cphi4'):
                    nvar += 1
        return nvar
