#!/home/lacerda/anaconda2/bin/python
import os
import sys
import pickle
import itertools
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib.lines import Line2D
from pytu.functions import debug_var
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from pytu.objects import readFileArgumentParser
from pytu.plots import plot_text_ax
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['legend.numpoints'] = 1
_transp_choice = False
# morphology
morph_name = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'I', 'BCD']
# morph_name = {'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6,
#               'E7': 7, 'S0': 8, 'S0a': 9, 'Sa': 10, 'Sab': 11, 'Sb': 12,
#               'Sbc': 13, 'Sc': 14, 'Scd': 15, 'Sd': 16, 'Sdm': 17, 'I': 18,
#               'BCD': 19}
# morph_name = ['E0','E1','E2','E3','E4','E5','E6','E7','S0','S0a','Sa','Sab','Sb', 'Sbc','Sc','Scd','Sd','Sdm','I','BCD']
latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)
color_AGN_tI = 'k'
color_AGN_tII = 'mediumslateblue'
scatter_kwargs = dict(s=2, vmax=14, vmin=3, cmap='viridis_r', marker='o', edgecolor='none')
scatter_AGN_tII_kwargs = dict(s=50, linewidth=0.1, marker='*', facecolor='none', edgecolor=color_AGN_tII)
scatter_AGN_tI_kwargs = dict(s=50, linewidth=0.1, marker='*', facecolor='none', edgecolor=color_AGN_tI)


def parser_args(default_args_file='args/default_plots.args'):
    """
    Parse the command line args.

    With fromfile_pidxrefix_chars=@ we can read and parse command line args
    inside a file with @file.txt.
    default args inside default_args_file
    """
    default_args = {
        'input': 'elines.pkl',
        'sigma_clip': True,
        'broad_fit_rules': False,
        'figs_dir': 'figs',
        'bug': 0.8,
        'img_suffix': 'pdf',
        'EW_SF': 14,
        'EW_hDIG': 3,
        'EW_strong': 6,
        'EW_verystrong': 10,
        'dpi': 300,
        'fontsize': 6,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input', '-I', metavar='FILE', type=str, default=default_args['input'])
    parser.add_argument('--sigma_clip', action='store_false', default=default_args['sigma_clip'])
    parser.add_argument('--figs_dir', '-D', metavar='DIR', type=str, default=default_args['figs_dir'])
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--img_suffix', '-i', metavar='IMG_SUFFIX', type=str, default=default_args['img_suffix'])
    parser.add_argument('--EW_SF', metavar='FLOAT', type=float, default=default_args['EW_SF'])
    parser.add_argument('--EW_hDIG', metavar='FLOAT', type=float, default=default_args['EW_hDIG'])
    parser.add_argument('--EW_strong', metavar='FLOAT', type=float, default=default_args['EW_strong'])
    parser.add_argument('--EW_verystrong', metavar='FLOAT', type=float, default=default_args['EW_verystrong'])
    parser.add_argument('--bug', metavar='FLOAT', type=float, default=default_args['bug'])
    parser.add_argument('--dpi', metavar='INT', type=int, default=default_args['dpi'])
    parser.add_argument('--fontsize', metavar='INT', type=int, default=default_args['fontsize'])
    args_list = sys.argv[1:]
    # if exists file default.args, load default args
    if os.path.isfile(default_args_file):
        args_list.insert(0, '@%s' % default_args_file)
    debug_var(True, args_list=args_list)
    args = parser.parse_args(args=args_list)
    args = parser.parse_args(args=args_list)
    debug_var(True, args=args)
    return args


def morph_adjust(x):
    r = x
    # If not NAN or M_TYPE E* (0-7) call it E
    if ~np.isnan(x) and x <= 7:
        r = 7
    return r


def modlogOHSF2017(x, a, b, c):
    return a + b * (x - c) * np.exp(c - x)


def modlogOHSF2017_t2(x):
    a = 8.85
    b = 0.007
    c = 11.5
    return modlogOHSF2017(x, a, b, c)


def count_y_above_mean(x, y, y_mean, x_bins):
    idx = np.digitize(x, x_bins)
    above = 0
    for i in np.unique(idx):
        above += (y[idx == i] > y_mean[i - 1]).astype('int').sum()
    return above


def xyz_clean_sort_interval(x, y, z=None, interval=None):
    if z is None:
        m = ~(np.isnan(x) | np.isnan(y))
    else:
        m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if interval is not None:
        m &= (x > interval[0]) & (x < interval[1]) & (y > interval[2]) & (y < interval[3])
    X = x[m]
    Y = y[m]
    iS = np.argsort(X)
    XS = X[iS]
    YS = Y[iS]
    if z is not None:
        Z = z[m]
        ZS = Z[iS]
        return XS, YS, ZS
    else:
        return XS, YS


def mean_xy_bins_interval(x, y, bins__r, interval=None, clip=None, mode='mean'):
    def redf(func, x, fill_value):
        if x.size == 0: return fill_value, 0
        if x.ndim == 1: return func(x), len(x)
        return func(x, axis=-1), x.shape[-1]
    if mode == 'mean': reduce_func = np.mean
    elif mode == 'median': reduce_func = np.median
    elif mode == 'sum': reduce_func = np.sum
    elif mode == 'var': reduce_func = np.var
    elif mode == 'std': reduce_func = np.std
    else: raise ValueError('Invalid mode: %s' % mode)
    nbins = len(bins__r) - 1
    idx = np.digitize(x, bins__r)
    y__R = np.ma.empty((nbins,), dtype='float')
    N__R = np.empty((nbins,))
    y_idxs = np.arange(0, y.shape[-1], dtype='int')
    if clip is not None:
        sel = []
        y_c__R = np.ma.empty((nbins,), dtype='float')
        y_sigma__R = np.ma.empty((nbins,), dtype='float')
        N_c__R = np.empty((nbins,))
        for i in range(0, nbins):
            y_bin = y[idx == i+1]
            y__R[i], N__R[i] = redf(np.mean, y_bin, np.ma.masked)
            delta = y_bin - y__R[i]
            y_sigma__R[i] = delta.std()
            m = np.abs(delta) < clip*y_sigma__R[i]
            y_bin = y_bin[m]
            sel = np.append(np.unique(sel), np.unique(y_idxs[idx == i+1][m]))
            y_c__R[i], N_c__R[i] = redf(reduce_func, y_bin, np.ma.masked)
        return y_c__R, N_c__R, y__R, N__R, sel.astype('int').tolist()
    else:
        sel = []
        for i in range(0, nbins):
            y_bin = y[idx == i+1]
            y__R[i], N__R[i] = redf(reduce_func, y_bin, np.ma.masked)
            sel = np.append(np.unique(sel), np.unique(y_idxs[idx == i+1]))
        return y__R, N__R, sel.astype('int').tolist()


def plot_setup(width, aspect, fignum=None, dpi=300, cmap=None):
    if cmap is None:
        cmap = 'inferno_r'
    plotpars = {
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'font.size': 8,
        'axes.titlesize': 10,
        'lines.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.8,
        'font.family': 'Times New Roman',
        'figure.subplot.left': 0.04,
        'figure.subplot.bottom': 0.04,
        'figure.subplot.right': 0.97,
        'figure.subplot.top': 0.95,
        'figure.subplot.wspace': 0.1,
        'figure.subplot.hspace': 0.25,
        'image.cmap': cmap,
    }
    plt.rcParams.update(plotpars)
    figsize = (width, width * aspect)
    return plt.figure(fignum, figsize, dpi=dpi)


def plot_colored_by_EW(elines, x, y, z, xlabel=None, ylabel=None, extent=None, n_bins_maj_x=5, n_bins_maj_y=5, n_bins_min_x=5, n_bins_min_y=5, prune_x='upper', prune_y=None, verbose=0, output_name=None, markAGNs=False, f=None, ax=None):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    if f is None:
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
    ax.scatter(x, y, c=z, **scatter_kwargs)
    if markAGNs:
        ax.scatter(x[mtII], y[mtII], **scatter_AGN_tII_kwargs)
        ax.scatter(x[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=args.fontsize+1)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=args.fontsize+1)
    cb_width = 0.05
    cb_ax = f.add_axes([right, bottom, cb_width, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'${\rm W}_{{\rm H}\alpha}$', fontsize=args.fontsize+1)
    cb.locator = MaxNLocator(2)
    # cb_ax.minorticks_on()
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    if extent is not None:
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(n_bins_maj_x, prune=prune_x))
    ax.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n_bins_min_x))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(**tick_params)
    ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    if verbose > 0:
        print '# x #'
        xlim = ax.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
        if verbose > 1:
            for i in x_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
        if verbose > 1:
            for i in x_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '#####'
        print '# y #'
        ylim = ax.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
        if verbose > 1:
            for i in y_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
        if verbose > 1:
            for i in y_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '#####'
    if output_name is not None:
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
    return f, ax


def plot_histo_xy_colored_by_EW(elines, x, y, z, ax_Hx, ax_Hy, ax_sc, xlabel=None, xrange=None, n_bins_maj_x=5, n_bins_min_x=5, prune_x=None, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, verbose=0):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    ax_Hx.hist(x, bins=15, range=xrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hx.hist(x[mtI], bins=15, range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hx.hist(x[mtII], bins=15, hatch='//////', range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    ax_Hx.set_xlabel(xlabel)
    ax_Hx.set_xlim(xrange)
    ax_Hx.xaxis.set_major_locator(MaxNLocator(n_bins_maj_x, prune=prune_x))
    ax_Hx.xaxis.set_minor_locator(AutoMinorLocator(n_bins_min_x))
    # ax_Hx.set_ylim(0, 1)
    ax_Hx.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax_Hx.yaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    ax_Hx.tick_params(**tick_params)
    ####################################
    ax_Hy.hist(y, orientation='horizontal', bins=15, range=yrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hy.hist(y[mtI], orientation='horizontal', bins=15, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hy.hist(y[mtII], orientation='horizontal', bins=15, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    ax_Hy.set_ylabel(ylabel)
    ax_Hy.set_ylim(yrange)
    ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    # ax_Hy.set_xlim(0, 1)
    ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=False, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    ax_Hy.tick_params(**tick_params)
    ####################################
    sc = ax_sc.scatter(x, y, c=z, **scatter_kwargs)
    ax_sc.scatter(x[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax_sc.scatter(x[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax_sc.set_xlim(ax_Hx.get_xlim())
    ax_sc.xaxis.set_major_locator(ax_Hx.xaxis.get_major_locator())
    ax_sc.xaxis.set_minor_locator(ax_Hx.xaxis.get_minor_locator())
    ax_sc.set_ylim(ax_Hy.get_ylim())
    ax_sc.yaxis.set_major_locator(ax_Hy.yaxis.get_major_locator())
    ax_sc.yaxis.set_minor_locator(ax_Hy.yaxis.get_minor_locator())
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax_sc.tick_params(**tick_params)
    ####################################
    pos = ax_sc.get_position()
    cb_width = 0.05
    cb_ax = f.add_axes([pos.x1, pos.y0, cb_width, pos.y1-pos.y0])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'${\rm W}_{{\rm H}\alpha}$', fontsize=args.fontsize+2)
    cb.locator = MaxNLocator(2)
    # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    ####################################
    if verbose > 0:
        print '# x #'
        xlim = ax_sc.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
        if verbose > 1:
            for i in x_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
        if verbose > 1:
            for i in x_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '#####'
        print '# y #'
        ylim = ax_sc.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
        if verbose > 1:
            for i in y_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
        if verbose > 1:
            for i in y_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '#####'
    return ax_Hx, ax_Hy, ax_sc


def plot_x_morph(elines, ax, verbose):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    x = elines['morph'].apply(morph_adjust)
    ax_Hx.hist(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hx.hist(x[mtI], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hx.hist(x[mtII], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], hatch='//////', histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    # ax.hist(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True)
    # ax.hist(x[mtII], hatch='//', bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tII, align='mid', rwidth=1, density=True)
    # ax.hist(x[mtI], hatch='////', bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tI, align='mid', rwidth=1, density=True)
    ax.set_xlabel(r'morphology')
    ax.set_xlim(6.5, 19.5)
    ticks = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ax.set_xticks(ticks)
    ax.set_xticklabels([morph_name[tick] for tick in ticks], rotation=90)
    ax.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    ax.tick_params(**tick_params)
    ####################################
    if verbose > 0:
        print '# x #'
        xlim = ax.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
        if verbose > 1:
            for i in x_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
        print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
        if verbose > 1:
            for i in x_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
        print '#####'
    return ax


def plot_morph_y_colored_by_EW(elines, y, ax_Hx, ax_Hy, ax_sc, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, verbose=0):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    EW_color = elines['EW_Ha_cen_mean'].apply(np.abs)
    morph = elines['morph'].apply(morph_adjust)
    scatter_kwargs = dict(c=EW_color, s=2, vmax=14, vmin=3, cmap='viridis_r', marker='o', edgecolor='none')
    ax_Hy.hist(y, orientation='horizontal', bins=15, range=yrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
    m = np.linspace(7, 19, 13).astype('int')
    y_mean = np.array([y.loc[morph == mt].mean() for mt in m])
    ax_Hy.hist(y[mtI], orientation='horizontal', bins=15, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    N_y_tI_above = np.array([np.array(y.loc[mtI & (morph == mt)] > y.loc[mtI & (morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
    print '# Type-I AGN above mean: %d (%.1f%%)' % (N_y_tI_above, 100.*N_y_tI_above/y[mtI].count())
    ax_Hy.hist(y[mtII], orientation='horizontal', bins=15, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    N_y_tII_above = np.array([np.array(y.loc[mtII & (morph == mt)] > y.loc[mtII & (morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
    print '# Type-II AGN above mean: %d (%.1f%%)' % (N_y_tII_above, 100.*N_y_tII_above/y[mtII].count())
    ax_Hy.set_ylabel(ylabel)
    ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_Hy.set_ylim(yrange)
    ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    tick_params = dict(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=False, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    ax_Hy.tick_params(**tick_params)
    ####################################
    sc = ax_sc.scatter(morph, y, **scatter_kwargs)
    ax_sc.scatter(morph[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax_sc.scatter(morph[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax_sc.plot(m, y_mean, 'k--')
    ax_sc.set_xlim(ax_Hx.get_xlim())
    ax_sc.xaxis.set_major_locator(ax_Hx.xaxis.get_major_locator())
    ax_sc.xaxis.set_minor_locator(ax_Hx.xaxis.get_minor_locator())
    ax_sc.set_ylim(ax_Hy.get_ylim())
    ax_sc.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax_sc.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax_sc.tick_params(**tick_params)
    ####################################
    pos = ax_sc.get_position()
    cb_width = 0.05
    cb_ax = f.add_axes([pos.x1, pos.y0, cb_width, pos.y1-pos.y0])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'${\rm W}_{{\rm H}\alpha}$', fontsize=args.fontsize+2)
    cb.locator = MaxNLocator(2)
    # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    ####################################
    if verbose > 0:
        print '# y #'
        ylim = ax_sc.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
        if verbose > 1:
            for i in y_low.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
        if verbose > 1:
            for i in y_upp.index:
                print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
        print '#####'
    return ax_Hx, ax_Hy, ax_sc


if __name__ == '__main__':
    args = parser_args()

    with open(args.input, 'rb') as f:
        pickled = pickle.load(f)

    elines = pickled['df']
    sel_NIIHa = pickled['sel_NIIHa']
    sel_OIIIHb = pickled['sel_OIIIHb']
    sel_SIIHa = pickled['sel_SIIHa']
    sel_OIHa = pickled['sel_OIHa']
    sel_EW = pickled['sel_EW']
    sel_AGNLINER_NIIHa_OIIIHb = pickled['sel_AGNLINER_NIIHa_OIIIHb']
    sel_SF_NIIHa_OIIIHb_K01 = pickled['sel_SF_NIIHa_OIIIHb_K01']
    sel_SF_NIIHa_OIIIHb_K03 = pickled['sel_SF_NIIHa_OIIIHb_K03']
    sel_SF_NIIHa_OIIIHb_S06 = pickled['sel_SF_NIIHa_OIIIHb_S06']
    sel_AGN_SIIHa_OIIIHb_K01 = pickled['sel_AGN_SIIHa_OIIIHb_K01']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01']
    sel_AGN_OIHa_OIIIHb_K01 = pickled['sel_AGN_OIHa_OIIIHb_K01']
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01']
    sel_AGN_candidates = pickled['sel_AGN_candidates']
    sel_SAGN_candidates = pickled['sel_SAGN_candidates']
    sel_VSAGN_candidates = pickled['sel_VSAGN_candidates']
    sel_pAGB = pickled['sel_pAGB']
    sel_SF_EW = pickled['sel_SF_EW']
    ###############################################################
    log_NII_Ha_cen = elines['log_NII_Ha_cen_fit']
    elog_NII_Ha_cen = elines['log_NII_Ha_cen_stddev']
    log_SII_Ha_cen = elines['log_SII_Ha_cen_mean']
    elog_SII_Ha_cen = elines['log_SII_Ha_cen_stddev']
    log_OI_Ha_cen = elines['log_OI_Ha_cen']
    elog_OI_Ha_cen = elines['e_log_OI_Ha_cen']
    log_OIII_Hb_cen = elines['log_OIII_Hb_cen_mean']
    elog_OIII_Hb_cen = elines['log_OIII_Hb_cen_stddev']
    EW_Ha_cen = elines['EW_Ha_cen_mean']
    eEW_Ha_cen = elines['EW_Ha_cen_stddev']
    ###############################################################
    L = Lines()
    consts_K01 = L.consts['K01']
    consts_K01_SII_Ha = L.consts['K01_SII_Ha']
    consts_K01_OI_Ha = L.consts['K01_OI_Ha']
    consts_K03 = L.consts['K03']
    consts_S06 = L.consts['S06']
    if args.sigma_clip:
        consts_K01 = L.sigma_clip_consts['K01']
        consts_K01_SII_Ha = L.sigma_clip_consts['K01_SII_Ha']
    ###############################################################

    mtI = elines['AGN_FLAG'] == 1
    N_AGN_tI = mtI.astype('int').sum()
    mtII = elines['AGN_FLAG'] == 2
    N_AGN_tII = mtII.astype('int').sum()
    legend_elements = [
        Line2D([0], [0], marker='*', markeredgecolor=color_AGN_tI, label='Type-I AGN (%d)' % N_AGN_tI, markerfacecolor='none', markersize=7, markeredgewidth=0.12, linewidth=0),
        Line2D([0], [0], marker='*', markeredgecolor=color_AGN_tII, label='Type-II AGN (%d)' % N_AGN_tII, markerfacecolor='none', markersize=7, markeredgewidth=0.12, linewidth=0),
    ]

    ##########################
    ## BPT colored by EW_Ha ##
    ##########################
    print '\n##########################'
    print '## BPT colored by EW_Ha ##'
    print '##########################'
    f = plot_setup(width=latex_text_width, aspect=1/3.)
    N_rows, N_cols = 1, 3
    bottom, top, left, right = 0.18, 0.95, 0.08, 0.9
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    fs = args.fontsize + 1
    y = log_OIII_Hb_cen
    ##########################
    ### NII/Ha
    print '##########################'
    print '## [NII]/Ha             ##'
    print '##########################'
    ax = ax0
    x = log_NII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen, **scatter_kwargs)
    ax.scatter(x.loc[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax.plot(L.x['K01'], L.y['K01'], 'k--')
    ax.plot(L.x['S06'], L.y['S06'], 'k-.')
    ax.plot(L.x['K03'], L.y['K03'], 'k-')
    ax.set_xlabel(r'$\log\ ({\rm [NII]}/{\rm H\alpha})$', fontsize=fs+4)
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN/LINER', 0.9, 0.95, fs+2, 'top', 'right', 'k')
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(**tick_params)
    ax.legend(handles=legend_elements, loc=2, frameon=False, fontsize='x-small', borderpad=0, borderaxespad=1)
    ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    ##########################
    # SII/Ha
    ##########################
    print '##########################'
    print '## [SII]/Ha             ##'
    print '##########################'
    ax = ax1
    x = log_SII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen, **scatter_kwargs)
    ax.scatter(x.loc[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax.set_xlabel(r'$\log\ ({\rm [SII]}/{\rm H\alpha})$', fontsize=fs+4)
    ax.plot(L.x['K01_SII_Ha'], L.y['K01_SII_Ha'], 'k--')
    ax.plot(L.x['K06_SII_Ha'], L.y['K06_SII_Ha'], 'k-.')
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN', 0.65, 0.95, fs+2, 'top', 'right', 'k')
    plot_text_ax(ax, 'LINER', 0.95, 0.85, fs+2, 'top', 'right', 'k')
    tick_params['labelleft'] = False
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(**tick_params)
    ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    ##########################
    # OI/Ha
    ##########################
    print '##########################'
    print '## [OI]/Ha              ##'
    print '##########################'
    ax = ax2
    x = log_OI_Ha_cen
    extent = [-3, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen, **scatter_kwargs)
    ax.scatter(x.loc[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax.set_xlabel(r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', fontsize=fs+4)
    cb_ax = f.add_axes([right, bottom, 0.02, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'${\rm W}_{{\rm H}\alpha}$', fontsize=fs+4)
    cb_ax.tick_params(direction='in')
    cb.locator = MaxNLocator(2)
    cb.update_ticks()
    ax.plot(L.x['K01_OI_Ha'], L.y['K01_OI_Ha'], 'k--')
    ax.plot(L.x['K06_OI_Ha'], L.y['K06_OI_Ha'], 'k-.')
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN', 0.65, 0.95, fs+2, 'top', 'right', 'k')
    plot_text_ax(ax, 'LINER', 0.95, 0.85, fs+2, 'top', 'right', 'k')
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(**tick_params)
    ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    ##########################
    f.text(0.01, 0.5, r'$\log\ ({\rm [OIII]}/{\rm H\beta})$', va='center', rotation='vertical', fontsize=fs+4)
    f.savefig('%s/fig_BPT.%s' % (args.figs_dir, args.img_suffix), dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print '##########################\n'
    ##########################

    ###########################
    ## SFMS colored by EW_Ha ##
    ###########################
    print '\n###########################'
    print '## SFMS colored by EW_Ha ##'
    print '###########################'
    x = elines['log_Mass']
    xlabel = r'$\log ({\rm M}_\star/{\rm M}_{\odot})$'
    extent = [8, 12.5, -4.5, 2.5]
    n_bins_min_x = 2
    n_bins_maj_y = 4
    n_bins_min_y = 2
    prune_x = None
    plot_colored_by_EW(elines=elines, x=x, y=elines['lSFR'], z=EW_Ha_cen, markAGNs=True,
                       ylabel=r'$\log ({\rm SFR}/{\rm M}_{\odot}/{\rm yr})$',
                       xlabel=xlabel, extent=extent,
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=args.verbose,
                       output_name='%s/fig_SFMS.%s' % (args.figs_dir, args.img_suffix))
    print '###########################\n'
    print '\n####################################'
    print '## SFMS colored by EW_Ha (NO CEN) ##'
    print '####################################'
    plot_colored_by_EW(elines=elines, x=x, y=elines['lSFR_NO_CEN'], z=EW_Ha_cen, markAGNs=True,
                       ylabel=r'$\log ({\rm SFR}/{\rm M}_{\odot}/{\rm yr})_{NO CEN}$',
                       xlabel=xlabel, extent=extent,
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=args.verbose,
                       output_name='%s/fig_SFMS_NC.%s' % (args.figs_dir, args.img_suffix))
    print '####################################\n'
    ################################

    ##########################
    ## M-C colored by EW_Ha ##
    ##########################
    print '\n##########################'
    print '## M-C colored by EW_Ha ##'
    print '##########################'
    n_bins_min_x = 2
    n_bins_maj_y = 6
    n_bins_min_y = 2
    plot_colored_by_EW(elines=elines, x=elines['log_Mass'], y=elines['C'], z=EW_Ha_cen, markAGNs=True,
                       xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                       ylabel=r'$\log ({\rm R}90/{\rm R}50)$',
                       extent=[8, 12.5, 0.5, 5.5],
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=args.verbose,
                       output_name='%s/fig_M_C.%s' % (args.figs_dir, args.img_suffix))
    print '##########################\n'
    ##########################

    #############################
    ## sSFR-C colored by EW_Ha ##
    #############################
    print '\n#############################'
    print '## sSFR-C colored by EW_Ha ##'
    print '#############################'
    n_bins_min_x = 2
    n_bins_maj_y = 6
    n_bins_min_y = 2
    output_name = '%s/fig_sSFR_C.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_EW(elines=elines, f=f, ax=ax, x=elines['lSFR'] - elines['log_Mass'], y=elines['C'], z=EW_Ha_cen,
                               xlabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                               ylabel=r'$\log ({\rm R}90/{\rm R}50)$',
                               extent=[-13.5, -8.5, 0.5, 5.5], markAGNs=True,
                               n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                               n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                               verbose=args.verbose)
    ax.axvline(x=-11.8, c='k', ls='--')
    ax.axvline(x=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print '#############################\n'
    ##########################


    #############################
    ## M-sSFR colored by EW_Ha ##
    #############################
    print '\n#############################'
    print '## M-sSFR colored by EW_Ha ##'
    print '#############################'
    n_bins_min_x = 2
    n_bins_maj_y = 5
    n_bins_min_y = 2
    output_name = '%s/fig_M_sSFR.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_EW(elines=elines, f=f, ax=ax, y=elines['lSFR'] - elines['log_Mass'], x=elines['log_Mass'], z=EW_Ha_cen,
                               ylabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                               xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                               extent=[8, 12.5, -13.5, -8.5], markAGNs=True,
                               n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                               n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                               verbose=args.verbose)
    ax.axhline(y=-11.8, c='k', ls='--')
    ax.axhline(y=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print '#############################\n'
    ##########################

    ################################
    ## X Y histo colored by EW_Ha ##
    ################################
    print '\n################################'
    print '## X Y histo colored by EW_Ha ##'
    print '################################'
    # print elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean']
    # print np.log10(np.abs(elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean']))
    m_redshift = (elines['redshift'] > 1e-6) & (elines['redshift'] < 0.2)
    EW_Ha_cen_zcut = elines.loc[m_redshift, 'EW_Ha_cen_mean'].apply(np.abs)
    EW_color_zcut = EW_Ha_cen_zcut.apply(np.log10)
    plot_histo_xy_dict = {
        ################################
        ## CMD (NSA) colored by EW_Ha ##
        ################################
        'fig_histo_CMD_NSA': [elines['Mabs_R'], r'${\rm M}_{\rm R}$ (mag)', 5, 2, None, elines['B_R'], r'${\rm B-R}$ (mag)', 3, 5, None, EW_Ha_cen, [-24, -10, 0, 1.5]],
        ################################

        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_CUBES': [elines.loc[m_redshift, 'Mabs_i'], r'${\rm M}_{\rm i}$ (mag)', 5, 2, None, elines.loc[m_redshift, 'u'] - elines.loc[m_redshift, 'i'], r'${\rm u}-{\rm i}$ (mag)', 3, 5, None, EW_color_zcut, [-24, -10, 0, 3.5]],
        ##################################

        ###########################
        ## SFMS colored by EW_Ha ##
        ###########################
        'fig_histo_SFMS': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', 4, 2, None, EW_Ha_cen, [8, 12.5, -4.5, 2.5]],
        ###########################

        ####################################
        ## SFMS colored by EW_Ha (NO CEN) ##
        ####################################
        'fig_histo_SFMS_NC': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR_NO_CEN'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})_{NO CEN}$', 4, 2, None, EW_Ha_cen, [8, 12.5, -4.5, 2.5]],
        ####################################

        ##########################
        ## M-C colored by EW_Ha ##
        ##########################
        'fig_histo_M_C': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['C'], r'$\log ({\rm R}90/{\rm R}50)$', 6, 2, None, EW_Ha_cen, [8, 12.5, 0.5, 5.5]],
        ##########################

        #############################
        ## sSFR-C colored by EW_Ha ##
        #############################
        'fig_histo_sSFR_C': [elines['lSFR'] - elines['log_Mass'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None, elines['C'], r'$\log ({\rm R}90/{\rm R}50)$', 6, 2, None, EW_Ha_cen, [-13.5, -8.5, 0.5, 5.5]],
        #############################

        #############################
        ## M-sSFR colored by EW_Ha ##
        #############################
        'fig_histo_M_sSFR': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR'] - elines['log_Mass'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None, EW_Ha_cen, [8, 12.5, -13.5, -8.5]],
        #############################

        ###############################
        ## M-ZHLWRe colored by EW_Ha ##
        ###############################
        'fig_histo_M_ZHLW': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['ZH_LW_Re_fit'], r'[Z/H] LW', 3, 2, None, EW_Ha_cen, [8, 12.5, -0.7, 0.3]],
        ###############################

        ###############################
        ## M-ZHMWRe colored by EW_Ha ##
        ###############################
        'fig_histo_M_ZHMW': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['ZH_MW_Re_fit'], r'[Z/H] LW', 3, 2, None, EW_Ha_cen, [8, 12.5, -0.7, 0.3]],
        ###############################

        ##########################
        ## MZR colored by EW_Ha ##
        ##########################
        'fig_histo_MZR': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['OH_Re_fit_t2'], r'$12 + \log (O/H)$ t2 ', 2, 5, None, EW_Ha_cen, [8, 12.5, 8.3, 9.1]],
        ###############################

        ############################
        ## M-tLW colored by EW_Ha ##
        ############################
        'fig_histo_M_tLW': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['Age_LW_Re_fit'], r'$\log({\rm age/yr})$ LW', 4, 5, None, EW_Ha_cen, [8, 12.5, 7.5, 10.5]],
        ###############################

        ############################
        ## M-tMW colored by EW_Ha ##
        ############################
        'fig_histo_M_tMW': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['Age_MW_Re_fit'], r'$\log({\rm age/yr})$ MW', 4, 5, None, EW_Ha_cen, [8, 12.5, 8.5, 10.5]],
        ###############################
    }

    for k, v in plot_histo_xy_dict.iteritems():
        print '\n################################'
        print '# %s' % k
        x, xlabel, n_bins_maj_x, n_bins_min_x, prune_x = v[0:5]
        y, ylabel, n_bins_maj_y, n_bins_min_y, prune_y = v[5:10]
        extent = v[-1]
        z = v[-2]
        output_name = '%s/%s.%s' % (args.figs_dir, k, args.img_suffix)
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        plot_histo_xy_colored_by_EW(elines=elines, x=x, y=y, z=z, ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, xlabel=xlabel, xrange=extent[0:2], n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, ylabel=ylabel, yrange=extent[2:4], n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=prune_y, verbose=args.verbose)
        if k == 'fig_histo_sSFR_C':
            ax_sc.axvline(x=-11.8, c='k', ls='--')
            ax_sc.axvline(x=-10.8, c='k', ls='--')
        if k == 'fig_histo_M_sSFR':
            ax_sc.axhline(y=-11.8, c='k', ls='--')
            ax_sc.axhline(y=-10.8, c='k', ls='--')
        if k == 'fig_histo_MZR':
            ### MZR ###
            mXnotnan = ~np.isnan(x)
            X = x.loc[mXnotnan].values
            iS = np.argsort(X)
            XS = X[iS]
            elines['modlogOHSF2017_t2'] = modlogOHSF2017_t2(elines['log_Mass'])
            YS = (elines['modlogOHSF2017_t2'].loc[mXnotnan].values)[iS]
            mY = YS > 8.4
            mX = XS < 11.5
            ax_sc.plot(XS[mX & mY], YS[mX & mY], 'k-')
            # ### best-fit ###
            # mnotnan = ~(np.isnan(x) | np.isnan(y)) & (x < 11.5) & (x > 8.3)
            # XFIT = x.loc[mnotnan].values
            # iSFIT = np.argsort(XFIT)
            # XFITS = XFIT[iSFIT]
            # YFIT = y.loc[mnotnan].values
            # YFITS = YFIT[iSFIT]
            # popt, pcov = curve_fit(f=modlogOHSF2017, xdata=XFITS, ydata=YFITS, p0=[8.8, 0.015, 11.5], bounds=[[8.54, 0.005, 11.499], [9, 0.022, 11.501]])
            # ax_sc.plot(XFITS, modlogOHSF2017(XFITS, *popt), 'k--')
            # print 'a:%.2f b:%.4f c:%.1f' % (popt[0], popt[1], popt[2])
            ### Above ###
            m_y_tI_above = y.loc[mtI] > x.loc[mtI].apply(modlogOHSF2017_t2)
            m_y_tII_above = y.loc[mtII] > x.loc[mtII].apply(modlogOHSF2017_t2)
            print elines.loc[m_y_tI_above.index[m_y_tI_above], ['log_Mass', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']]
            print elines.loc[m_y_tII_above.index[m_y_tII_above], ['log_Mass', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']]
            N_y_tI_above = m_y_tI_above.astype('int').sum()
            N_y_tII_above = m_y_tII_above.astype('int').sum()
            print '# Type-I AGN above SF2017 curve: %d (%.1f%%)' % (N_y_tI_above, 100.*N_y_tI_above/y[mtI].count())
            print '# Type-II AGN above SF2017 curve: %d (%.1f%%)' % (N_y_tII_above, 100.*N_y_tII_above/y[mtII].count())
        if k == 'fig_histo_M_ZHMW':
            x_bins = np.arange(9-0.3, 11.5+0.3, 0.3)
            x_bincenter = (x_bins[:-1] + x_bins[1:]) / 2.0
            nbins = len(x_bincenter)
            interval = [8.7, 11.8, -0.7, 0.3]
            y_mean, N_y_mean = mean_xy_bins_interval(x.values, y.values, x_bins, interval)
            ax_sc.plot(x_bincenter, y_mean, 'k-')
            ### above ###
            x_AGNs_tI = x.loc[elines['AGN_FLAG'] == 1].values
            y_AGNs_tI = y.loc[elines['AGN_FLAG'] == 1].values
            N_y_tI_above = count_y_above_mean(x_AGNs_tI, y_AGNs_tI, y_mean, x_bins)
            x_AGNs_tII = x.loc[elines['AGN_FLAG'] == 2].values
            y_AGNs_tII = y.loc[elines['AGN_FLAG'] == 2].values
            N_y_tII_above = count_y_above_mean(x_AGNs_tII, y_AGNs_tII, y_mean, x_bins)
            print '# Type-I AGN above mean: %d (%.1f%%)' % (N_y_tI_above, 100.*N_y_tI_above/y[mtI].count())
            print '# Type-II AGN above mean: %d (%.1f%%)' % (N_y_tII_above, 100.*N_y_tII_above/y[mtII].count())
        if k == 'fig_histo_M_tLW' or k == 'fig_histo_M_tMW':
            WHa = elines['EW_Ha_ALL']
            y_AGNs_mean = y.loc[elines['AGN_FLAG'] > 0].mean()
            y_AGNs_tI_mean = y.loc[elines['AGN_FLAG'] == 1].mean()
            y_AGNs_tII_mean = y.loc[elines['AGN_FLAG'] == 2].mean()
            m = ~(np.isnan(x) | np.isnan(y) | np.isnan(WHa))
            x_SF = x.loc[m & (WHa > 14)]
            y_SF = y.loc[m & (WHa > 14)]
            y_SF_mean = y_SF.mean()
            x_hDIG = x.loc[m & (WHa <= 3)]
            y_hDIG = y.loc[m & (WHa <= 3)]
            y_hDIG_mean = y.loc[m & (WHa <= 3)].mean()
            print 'y_AGNs_mean: %.2f Gyr' % 10**(y_AGNs_mean - 9)
            print 'y_AGNs_tI_mean: %.2f Gyr' % 10**(y_AGNs_tI_mean - 9)
            print 'y_AGNs_tII_mean: %.2f Gyr' % 10**(y_AGNs_tII_mean - 9)
            print 'y_SF_mean: %.2f Gyr' % 10**(y_SF_mean - 9)
            print 'y_hDIG_mean: %.2f Gyr' % 10**(y_hDIG_mean - 9)
            ### MSFS ###
            SFRHa = elines['lSFR']
            m = ~(np.isnan(x) | np.isnan(y) | np.isnan(SFRHa))
            x_SF = x.loc[m & (WHa > 14)]
            y_SF = y.loc[m & (WHa > 14)]
            SFRHa_SF = SFRHa.loc[m & (WHa > 14)]
            iS = np.argsort(x_SF)
            XS_SF = x_SF[iS]
            YS_SF = y_SF[iS]
            SFRHaS_SF = SFRHa_SF[iS]
            x_bins = np.arange(8, 11.5, 0.3)
            aSF, bSF = np.polyfit(XS_SF, SFRHaS_SF, 1)
            print aSF, bSF
            ax_sc.plot(XS_SF, aSF * XS_SF + bSF, 'k--')
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print '################################\n'
    print '################################\n'
    ################################
    ################################

    ###########
    ## Morph ##
    ###########
    # Create an object spanning only galaxies with defined morphology
    elines_wmorph = elines.loc[elines['morph'] >= 0].copy()
    print '#################'
    print '## Morph plots ##'
    print '#################\n'
    ##########################
    ## Morph colored by EW_Ha ##
    ############################
    print '\n############################'
    print '## Morph colored by EW_Ha ##'
    print '############################'
    plots_dict = {
        'fig_Morph_M': ['log_Mass', [8, 12.5], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 6, 2],
        'fig_Morph_C': ['C', [0.5, 5.5], r'$\log ({\rm R}90/{\rm R}50)$', 6, 2],
        'fig_Morph_SigmaMassCen': ['Sigma_Mass_cen', [1, 5], r'$\log (\Sigma^\star/{\rm M}_{\odot}/{\rm pc}^{-2})$', 4, 2],
        'fig_Morph_vsigma': ['rat_vel_sigma', [0, 1], r'${\rm v}/\sigma ({\rm R} < {\rm Re})$', 2, 2],
        'fig_Morph_Re': ['Re_kpc', [0, 25], r'${\rm Re}/{\rm kpc}$', 6, 2]
    }
    for k, v in plots_dict.iteritems():
        print '\n############################'
        print '# %s ' % k
        ykey, yrange, ylabel, n_bins_maj_y, n_bins_min_y = v
        y = elines_wmorph[ykey]
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        output_name = '%s/%s.%s' % (args.figs_dir, k, args.img_suffix)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        ax_Hx = plot_x_morph(elines=elines_wmorph, ax=ax_Hx, verbose=args.verbose)
        plot_morph_y_colored_by_EW(elines=elines_wmorph, y=y, ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, ylabel=ylabel, yrange=yrange, n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=None, verbose=args.verbose)
        # gs.tight_layout(f)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print '############################\n'
    print '############################\n'
    ############################

###############################################################################
# END PLOTS ###################################################################
###############################################################################