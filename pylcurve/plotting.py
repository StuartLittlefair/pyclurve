import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner as triangle
import matplotlib as mpl
from astropy.table import Table
from .utils import t2phase


def plot_LC(model, params, show=True, save=False, name='LC.pdf', dataname=None):

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

    n_cols = len(model.lightcurves.keys())
    fig = plt.figure(figsize=(n_cols * 3, 4))
    gs = gridspec.GridSpec(4, n_cols, figure=fig, wspace=0, hspace=0)
    gs.update(wspace=0, hspace=0)
    t0_idx = list(model.config['params'].keys()).index('t0')
    for idx in range(n_cols*2):
        idx_half = int(idx/2)
        band = list(model.lightcurves.keys())[idx_half]
        t, ym, y, ye = model.model(band, params)
        if dataname:
            np.savetxt('{}_{}.dat'.format(dataname,band), np.column_stack((t,ym)))
        
        phase = t2phase(t, params[t0_idx], model.config['period'])

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
            ax.errorbar(phase, (y - ym) / np.std((y - ym)),
                        yerr=ye / np.std((y - ym)), lw=0, elinewidth=1,
                        marker='.', ms=2, zorder=1, color=band_colours[band])
            ax.axhline(0, c='k', ls='--', zorder=2)
            ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
            yabs_max_res = np.max(np.abs(ax.get_ylim()))
            ax.set_ylim([-4.0, 4.0])
            ax.set_yticks([-3.0, 0.0, 3.0])
            ax.set_yticklabels([r'$-3\sigma$', '0', r'$3\sigma$'])
        if idx_half != 0:
            plt.setp(ax.get_yticklabels(), visible=False)

    for axis, band in zip(fig.get_axes()[::2], list(model.lightcurves.keys())):
        axis.annotate(annotate_str[band], xy=(0, y_mid), color='k', fontsize=18)

    plt.figtext(0.06, 0.5, 'Flux (mJy)', rotation='vertical')
    plt.figtext(0.47, 0.015, r'Orbital phase, $\phi$')
    if save:
        plt.savefig(name, pad_inches=0.1, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_SED(model, params, show=True, save=False, name='SED.pdf'):

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(5, 1, figure=fig, wspace=0, hspace=0)
    gs.update(wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[:4, 0])
    ax1 = fig.add_subplot(gs[4:, 0], sharex=ax0)

    model.set_parameter_vector(params)
    wd_model = []
    wd_real = []
    wd_err = []
    wavelength = []
    for band in model.lightcurves.keys():
        ym, wdwarf, wd_model_flux = model.get_value(band)
        ax0.errorbar(model.cam.eff_wl[band].value, wdwarf*1e3,
                    yerr=model.flux_uncertainty[band]*wdwarf*1e3, c='k',
                    marker='.', elinewidth=1)
        ax0.scatter(model.cam.eff_wl[band].value, wd_model_flux*1e3, c='r', marker='.')
        ax1.errorbar(model.cam.eff_wl[band].value, wdwarf*1e3 - wd_model_flux*1e3,
                     yerr=model.flux_uncertainty[band]*wdwarf*1e3, c='k',
                     marker='.', elinewidth=1)
        ax1.axhline
        wd_model.append(wd_model_flux)
        wd_real.append(wdwarf)
        wd_err.append(model.flux_uncertainty[band]*wdwarf)
        wavelength.append(model.cam.eff_wl[band].value)
    ax1.axhline(0, c='k', ls='--', lw=1)
    ax0.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    ax1.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    ax0.set_ylim(bottom=0)
    ax0.set_xlim(3300, 9500)
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(-yabs_max, yabs_max)
    plt.setp(ax0.get_xticklabels(), visible=False)
    if save:
        plt.savefig(name)
    if show:
        plt.show()
    plt.close()
    t = Table()
    t['wavelength'], t['wdflux'], t['wdflux_e'], t['wdmodel'] = wavelength, wd_real, wd_err, wd_model
    t.write(f"MCMC_runs/{model.config['run_name']}/{model.config['run_name']}_SED.dat", format='ascii', overwrite=True)


def plot_CP(fchain, namelist, name='CP.pdf', **kwargs):
    mpl.rcParams['font.size'] = 12
    label_dict = {'t1': r'$\rm{T_{1}~(K)}$',
                  't2': r'$\rm{T_{2}~(K)}$',
                  'm1': r'$\rm{M_{1}~(M_{\odot})}$',
                  'm2': r'$\rm{M_{2}~(M_{\odot})}$',
                  'incl': r'$\rm{i~(deg)}$',
                  't0': r'$\rm{\Delta T_{0}~(10^{-5}~d})$',
                  'parallax': r'$\rm{\pi~(mas)}$',
                  'ebv': r'$E(B-V)$',
                  'ln_prob': r'$\rm{ln}(p)$'
                  }
    labels = [label_dict[name] if name in label_dict else name for name in namelist]
    idx_t0 = namelist.index('t0')
    fchain_plot = fchain.copy()
    fchain_plot[:, idx_t0] = (fchain_plot[:,idx_t0] - np.median(fchain_plot[:,idx_t0])) * 1e5
    fig = triangle.corner(fchain_plot, labels=labels, hist_bin_factor=2, label_kwargs={'fontsize': 14}, **kwargs)
    fig.savefig(name)
    plt.close()


def plot_traces(chain, namelist, name="trace.pdf", dpi=300, **kwargs):
    mpl.rcParams['font.size'] = 8
    label_dict = {'t1': r'$\rm{T_{1}~(K)}$',
                  't2': r'$\rm{T_{2}~(K)}$',
                  'm1': r'$\rm{M_{1}~(M_{\odot})}$',
                  'm2': r'$\rm{M_{2}~(M_{\odot})}$',
                  'incl': r'$\rm{i~(deg)}$',
                  't0': r'$\rm{\Delta T_{0}~(10^{-5}~d})$',
                  'parallax': r'$\rm{\pi~(mas)}$',
                  'ebv': r'$E(B-V)$',
                  'ln_prob': r'$\rm{ln}(p)$'
                  }

    npars = len(namelist)
    fig, axes = plt.subplots(npars, 1, figsize=(8, npars), sharex=True)
    nsteps, nwalkers, npars = chain.shape
    for npar, ax in enumerate(axes):
        if namelist[npar] == 't0':
            ax.plot((chain[:, :, npar] - np.median(chain[:,:,npar]))*1e5, color='k', alpha=0.1, **kwargs)
        else:
            ax.plot(chain[:, :, npar], color='k', alpha=0.1, **kwargs)
        if namelist[npar] in label_dict:
            ax.set_ylabel(label_dict[namelist[npar]])
        else:
            ax.set_ylabel(namelist[npar])
        if npar != len(axes)-1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    axes[-1].set_xlabel('step number')
    plt.subplots_adjust(hspace=0.05)
    fig.align_ylabels(axes)
    fig.savefig(name, pad_inches=0.1, bbox_inches='tight', dpi=dpi)
    plt.close()
