#!/usr/bin/python3
import os
import sys
import pickle
import itertools
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
import seaborn as sns
from pytu.plots import plot_text_ax
from matplotlib.lines import Line2D
from pytu.functions import debug_var
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from pytu.functions import ma_mask_xyz
from pytu.objects import readFileArgumentParser
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['legend.numpoints'] = 1
_transp_choice = False
# morphology
morph_name = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'I']
# morph_name = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'I']
morph_name_ticks = ['', 'E', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'I', '']
# morph_name = {'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6,
#               'E7': 7, 'S0': 8, 'S0a': 9, 'Sa': 10, 'Sab': 11, 'Sb': 12,
#               'Sbc': 13, 'Sc': 14, 'Scd': 15, 'Sd': 16, 'Sdm': 17, 'Sm': 18,
#               'I': 19}
# morph_name = ['E0','E1','E2','E3','E4','E5','E6','E7','S0','S0a','Sa','Sab','Sb', 'Sbc','Sc','Scd','Sd','Sdm','Sm','I']
latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)
color_all = 'green'
color_AGN_tI = 'k'
color_AGN_tII = 'blue'
color_AGN_tIII = 'red'
color_AGN_tIV = 'red'
marker_AGN_tI = '*'
marker_AGN_tII = '*'
marker_AGN_tIII = 'o'
marker_AGN_tIV = 'X'
alpha_AGN_tIII = 1
alpha_AGN_tIV = 1
scatter_kwargs = dict(s=1, cmap='viridis_r', marker='o', edgecolor='none', alpha=1)
scatter_kwargs_EWmaxmin = dict(s=1, vmax=2.5, vmin=-1, cmap='viridis_r', marker='o', edgecolor='none', alpha=1)
scatter_AGN_tIV_kwargs = dict(s=30, alpha=alpha_AGN_tIV, linewidth=0.5, marker=marker_AGN_tIV, facecolor='none', edgecolor=color_AGN_tIV)
scatter_AGN_tIII_kwargs = dict(s=30, alpha=alpha_AGN_tIII, linewidth=0.5, marker=marker_AGN_tIII, facecolor='none', edgecolor=color_AGN_tIII)
scatter_AGN_tII_kwargs = dict(s=30, linewidth=0.5, marker=marker_AGN_tII, facecolor='none', edgecolor=color_AGN_tII)
scatter_AGN_tI_kwargs = dict(s=30, linewidth=0.5, marker=marker_AGN_tI, facecolor='none', edgecolor=color_AGN_tI)
n_levels_kdeplot = 4


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
        'img_suffix': 'pdf',
        'dpi': 300,
        'fontsize': 6,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input', '-I', metavar='FILE', type=str, default=default_args['input'])
    parser.add_argument('--sigma_clip', action='store_false', default=default_args['sigma_clip'])
    parser.add_argument('--figs_dir', '-D', metavar='DIR', type=str, default=default_args['figs_dir'])
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--img_suffix', '-i', metavar='IMG_SUFFIX', type=str, default=default_args['img_suffix'])
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


def create_bins(interval, step):
    bins = np.arange(interval[0]-step, interval[1]+step, step)
    bins_center = (bins[:-1] + bins[1:])/2.
    return bins, bins_center, len(bins_center)


def count_y_above_mean(x, y, y_mean, x_bins, interval=None):
    x, y = xyz_clean_sort_interval(x, y, interval=interval)
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


def redf_xy_bins_interval(x, y, bins__r, interval=None, clip=None, mode='mean'):
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
    N__R = np.empty((nbins,), dtype='int')
    y_idxs = np.arange(0, y.shape[-1], dtype='int')
    if clip is not None:
        sel = []
        sel_c = []
        y_c__R = np.ma.empty((nbins,), dtype='float')
        y_sigma__R = np.ma.empty((nbins,), dtype='float')
        N_c__R = np.empty((nbins,), dtype='int')
        for i in range(0, nbins):
            y_bin = y[idx == i+1]
            y__R[i], N__R[i] = redf(np.mean, y_bin, np.ma.masked)
            sel = np.append(np.unique(sel), np.unique(y_idxs[idx == i+1]))
            if N__R[i] != 0:
                delta = y_bin - y__R[i]
                y_sigma__R[i] = delta.std()
                m = np.abs(delta) < clip*y_sigma__R[i]
                y_bin = y_bin[m]
                sel_c = np.append(np.unique(sel_c), np.unique(y_idxs[idx == i+1][m]))
                y_c__R[i], N_c__R[i] = redf(reduce_func, y_bin, np.ma.masked)
            else:
                y_c__R[i], N_c__R[i] = np.ma.masked, 0
            # print(sel_c)
            # print(sel)
        return y_c__R, N_c__R, sel_c.astype('int').tolist(), y__R, N__R, sel.astype('int').tolist()
    else:
        sel = []
        for i in range(0, nbins):
            y_bin = y[idx == i+1]
            y__R[i], N__R[i] = redf(reduce_func, y_bin, np.ma.masked)
            sel = np.append(np.unique(sel), np.unique(y_idxs[idx == i+1]))
        return None, None, None, y__R, N__R, sel.astype('int').tolist()


def linear_regression_mean(x, y, interval, step=None, clip=None):
    pc = None
    if step is None:
        step = 0.1
    XS, YS = xyz_clean_sort_interval(x.values, y.values)
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], step)
    YS_c__r, N_c__r, sel_c, YS__r, N__r, sel = redf_xy_bins_interval(XS, YS, x_bins__r, clip=clip, interval=interval)
    if clip is not None:
        xm, ym = ma_mask_xyz(x_bins_center__r, YS_c__r)
        pc = np.ma.polyfit(xm.compressed(), ym.compressed(), 1)
        print('linear regression with {:d} sigma clip:'.format(clip))
        print(pc)
        sigma_dev = (y - np.polyval(pc, x)).std()
        print('sigma dev = {:.5f}'.format(sigma_dev))
    xm, ym = ma_mask_xyz(x_bins_center__r, YS__r)
    p = np.ma.polyfit(xm.compressed(), ym.compressed(), 1)
    print('linear regression with no sigma clip:')
    print(p)
    sigma_dev = (y - np.polyval(p, x)).std()
    print('sigma dev = {:.5f}'.format(sigma_dev))
    return p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel


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


def plot_WHAN(args, N2Ha, WHa, z=None, f=None, ax=None, extent=None, output_name=None, cmap='viridis', mask=None, N=False, z_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True):
    from pytu.plots import plot_text_ax
    from pytu.functions import ma_mask_xyz

    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    if f is None:
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
    if mask is None:
        mask = np.zeros_like(N2Ha, dtype=np.bool_)
    if extent is None:
        extent = [-1.6, 0.8, -1, 2.5]
    if z is None:
        bins = [30, 30]
        xm, ym = ma_mask_xyz(N2Ha, np.ma.log10(WHa), mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax, range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, **scatter_kwargs)
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, np.ma.log10(WHa), z, mask=mask)
        #   print(xm, ym, z)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=1, edgecolor='none')
        cb_width = 0.05
        cb_ax = f.add_axes([right, bottom, cb_width, top-bottom])
        cb = plt.colorbar(sc, cax=cb_ax)
        cb.set_label(z_label, fontsize=args.fontsize+1)
        cb.locator = MaxNLocator(4)
        # cb_ax.minorticks_on()
        cb_ax.tick_params(which='both', direction='in')
        cb.update_ticks()
    xlabel = r'$\log\ ({\rm [NII]}/{\rm H\alpha})$'
    ylabel = r'$\log {\rm EW}({\rm H\alpha})$'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not N:
        N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plot_text_ax(ax, '%d %s' % (N, c), 0.02, 0.98, args.fontsize, 'top', 'left', 'k')
    ax.plot((-0.4, -0.4), (np.log10(3), 3), 'k-')
    ax.plot((-0.4, extent[1]), np.ma.log10([6, 6]), 'k-')
    ax.axhline(y=np.log10(3), c='k')
    p = [np.log10(0.5/5.0), np.log10(0.5)]
    xini = (np.log10(3.) - p[1]) / p[0]
    ax.plot((xini, 0.), np.polyval(p, [xini, 0.]), 'k:')
    ax.plot((0, extent[1]), np.log10([0.5, 0.5]), 'k:')
    ax.text(-1.4, 0.75, 'SF', fontsize=args.fontsize)
    ax.text(0.5, 2.25, 'sAGN', fontsize=args.fontsize)
    ax.text(0.8, 0.55, 'wAGN', fontsize=args.fontsize)
    ax.text(0.25, 0.0, 'RG', fontsize=args.fontsize)
    ax.text(-0.8, 0, 'PG', fontsize=args.fontsize)
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(**tick_params)
    if output_name is not None:
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
    else:
        return f, ax


def plot_colored_by_z(elines, args, x, y, z, xlabel=None, ylabel=None, z_label=None, extent=None, n_bins_maj_x=5, n_bins_maj_y=5, n_bins_min_x=5, n_bins_min_y=5, prune_x='upper', prune_y=None, output_name=None, markAGNs=False, f=None, ax=None, sc_kwargs=None):
    if z_label is None:
        z_label = r'${\rm W}_{{\rm H}\alpha}$'
    if sc_kwargs is None:
        sc_kwargs = scatter_kwargs_EWmaxmin
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    if f is None:
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
    sc = ax.scatter(x, y, c=z, **sc_kwargs)
    # mALLAGN = (elines['AGN_FLAG'] > 0)
    xm, ym = ma_mask_xyz(x, y, mask=~mtAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    if markAGNs:
        # ax.scatter(x[mtIII], y[mtIII], **scatter_AGN_tIII_kwargs)
        ax.scatter(x[mtII], y[mtII], **scatter_AGN_tII_kwargs)
        ax.scatter(x[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=args.fontsize+1)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=args.fontsize+1)
    cb_width = 0.05
    cb_ax = f.add_axes([right, bottom, cb_width, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(z_label, fontsize=args.fontsize+1)
    cb.locator = MaxNLocator(4)
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
    # ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    if args.verbose > 0:
        print('# x #')
        xlim = ax.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print('# N.x points < %.1f: %d' % (xlim[0], x_low.count()))
        if args.verbose > 1:
            for i in x_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.x points > %.1f: %d' % (xlim[1], x_upp.count()))
        if args.verbose > 1:
            for i in x_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
        print('# y #')
        ylim = ax.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print('# N.y points < %.1f: %d' % (ylim[0], y_low.count()))
        if args.verbose > 1:
            for i in y_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.y points > %.1f: %d' % (ylim[1], y_upp.count()))
        if args.verbose > 1:
            for i in y_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
    if output_name is not None:
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
    return f, ax


def plot_histo_xy_colored_by_z(elines, args, x, y, z, ax_Hx, ax_Hy, ax_sc, xlabel=None, xrange=None, n_bins_maj_x=5, n_bins_min_x=5, prune_x=None, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, aux_mask=None):
    if aux_mask is not None:
        elines = elines.loc[aux_mask]
    mtI = (elines['AGN_FLAG'] == 1)
    mtII = (elines['AGN_FLAG'] == 2)
    # mtIII = elines['AGN_FLAG'] == 3
    Nbins = 20
    mtAGN = mtI | mtII
    ax_Hx_t = ax_Hx.twinx()
    ax_Hx_t.hist(x, bins=Nbins, range=xrange, histtype='step', fill=True, facecolor=color_all, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hx.hist(x[mtI], bins=Nbins, range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hx.hist(x[mtII], bins=Nbins, hatch='//////', range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    # ax_Hx.hist(x[mtAGN], bins=Nbins, hatch='\\\\', range=xrange, histtype='step', linestyle='dashed', linewidth=1, edgecolor=color_AGN_tIII, align='mid', density=True)
    ax_Hx.hist(x[mtAGN], bins=Nbins, range=xrange, histtype='step', fill=True, facecolor=color_AGN_tIV, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hx.set_xlabel(xlabel)
    ax_Hx.set_xlim(xrange)
    ax_Hx.xaxis.set_major_locator(MaxNLocator(n_bins_maj_x, prune=prune_x))
    ax_Hx.xaxis.set_minor_locator(AutoMinorLocator(n_bins_min_x))
    # ax_Hx.set_ylim(0, 1)
    # ax_Hx.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    # ax_Hx.yaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    ax_Hx.tick_params(**tick_params)
    ax_Hx_t.tick_params(**tick_params)
    ####################################
    ax_Hy_t = ax_Hy.twiny()
    ax_Hy_t.hist(y, orientation='horizontal', bins=Nbins, range=yrange, histtype='step', fill=True, facecolor=color_all, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hy.hist(y[mtI], orientation='horizontal', bins=Nbins, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hy.hist(y[mtII], orientation='horizontal', bins=Nbins, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    # ax_Hy.hist(y[mtAGN], orientation='horizontal', hatch='\\\\', bins=Nbins, range=yrange, linestyle='dashed', histtype='step', linewidth=1, edgecolor=color_AGN_tIII, align='mid', density=True)
    ax_Hy.hist(y[mtAGN], orientation='horizontal', bins=Nbins, range=yrange, histtype='step', fill=True, facecolor=color_AGN_tIV, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hy.set_ylabel(ylabel)
    ax_Hy.set_ylim(yrange)
    ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    # ax_Hy.set_xlim(0, 1)
    # ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    # ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=False, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    ax_Hy.tick_params(**tick_params)
    ax_Hy_t.tick_params(**tick_params)
    ####################################
    # print(len(x), len(y), len(mtAGN))
    # xm, ym = ma_mask_xyz(x, y, mask=~mtAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax_sc, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    sc = ax_sc.scatter(x, y, c=z, **scatter_kwargs_EWmaxmin)
    # ax_sc.scatter(x[mtIII], y[mtIII], **scatter_AGN_tIII_kwargs)
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
    cb.set_label(r'$\log {\rm W}_{{\rm H}\alpha}$ (\AA)', fontsize=args.fontsize+2)
    cb.locator = MaxNLocator(4)
    # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    ####################################
    if args.verbose > 0:
        print('# x #')
        xlim = ax_sc.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print('# N.x points < %.1f: %d' % (xlim[0], x_low.count()))
        if args.verbose > 1:
            for i in x_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.x points > %.1f: %d' % (xlim[1], x_upp.count()))
        if args.verbose > 1:
            for i in x_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
        print('# y #')
        ylim = ax_sc.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print('# N.y points < %.1f: %d' % (ylim[0], y_low.count()))
        if args.verbose > 1:
            for i in y_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.y points > %.1f: %d' % (ylim[1], y_upp.count()))
        if args.verbose > 1:
            for i in y_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
    return ax_Hx, ax_Hy, ax_sc


def plot_x_morph(elines, args, ax):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    x = elines['morph'].apply(morph_adjust)
    ax_t = ax.twinx()
    ax_t.hist(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor=color_all, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax.hist(x[mtI], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax.hist(x[mtII], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], hatch='//////', histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    # ax.hist(x[mtAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], hatch='\\\\', histtype='step', linewidth=1, linestyle='dashed', edgecolor=color_AGN_tIII, align='mid', density=True)
    ax.hist(x[mtAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor=color_AGN_tIV, edgecolor='none', align='mid', density=True, alpha=0.5)
    # ax.hist(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor=color_all, edgecolor='none', align='mid', density=True)
    # ax.hist(x[mtII], hatch='//', bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tII, align='mid', rwidth=1, density=True)
    # ax.hist(x[mtI], hatch='////', bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tI, align='mid', rwidth=1, density=True)
    ax.set_xlabel(r'morphology')
    ax.set_xlim(5.5, 20.5)
    ticks = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ax.set_xticks(ticks)
    ax.set_xticklabels([morph_name_ticks[tick] for tick in (np.array(ticks, dtype='int') - 6)], rotation=90)
    ax.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    ax.tick_params(**tick_params)
    ax_t.tick_params(**tick_params)
    ####################################
    if args.verbose > 0:
        print('# x #')
        xlim = ax.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print('# N.x points < %.1f: %d' % (xlim[0], x_low.count()))
        if args.verbose > 1:
            for i in x_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines_wmorph.loc[i, 'AGN_FLAG']))
        print('# N.x points > %.1f: %d' % (xlim[1], x_upp.count()))
        if args.verbose > 1:
            for i in x_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines_wmorph.loc[i, 'AGN_FLAG']))
        print('#####')
    return ax


def plot_morph_y_colored_by_EW(elines, args, y, ax_Hx, ax_Hy, ax_sc, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None):
    EW_color = elines['EW_Ha_cen_mean'].apply(np.abs)
    scatter_kwargs_EWmaxmin = dict(c=EW_color.apply(np.log10), s=2, vmax=2.5, vmin=-1, cmap='viridis_r', marker='o', edgecolor='none')
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    morph = elines['morph'].apply(morph_adjust)
    m = np.linspace(7, 19, 13).astype('int')
    y_mean = np.array([y.loc[morph == mt].mean() for mt in m])
    N_y_tI = y[mtI].count()
    N_y_tII = y[mtII].count()
    N_y_tAGN = y[mtAGN].count()
    N_y_tI_above = count_y_above_mean(morph.loc[mtI], y.loc[mtI], y_mean, m)
    # np.array([np.array(y.loc[mtI & (morph == mt)] > y.loc[(morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
    N_y_tII_above = count_y_above_mean(morph.loc[mtII], y.loc[mtII], y_mean, m)
    # np.array([np.array(y.loc[mtII & (morph == mt)] > y.loc[(morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
    N_y_tAGN_above = count_y_above_mean(morph.loc[mtAGN], y.loc[mtAGN], y_mean, m)
    # np.array([np.array(y.loc[mtAGN & (morph == mt)] > y.loc[(morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
    Nbins = 20
    ax_Hy_t = ax_Hy.twiny()
    ax_Hy_t.hist(y, orientation='horizontal', bins=Nbins, range=yrange, histtype='step', fill=True, facecolor=color_all, edgecolor='none', align='mid', density=True, alpha=0.5)
    ax_Hy.hist(y[mtI], orientation='horizontal', bins=Nbins, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
    ax_Hy.hist(y[mtII], orientation='horizontal', bins=Nbins, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
    # ax_Hy.hist(y[mtAGN], hatch='\\\\', linestyle='dashed', orientation='horizontal', bins=Nbins, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tIII, align='mid', density=True)
    ax_Hy.hist(y[mtAGN], orientation='horizontal', bins=Nbins, range=yrange, histtype='step', fill=True, facecolor=color_AGN_tIV, edgecolor='none', align='mid', density=True, alpha=0.5)
    print('# B.F. Type-I AGN above mean: %d/%d (%.1f%%)' % (N_y_tI_above, N_y_tI, 100.*N_y_tI_above/N_y_tI))
    print('# B.F. Type-II AGN above mean: %d/%d (%.1f%%)' % (N_y_tII_above, N_y_tII, 100.*N_y_tII_above/N_y_tII))
    print('# ALL AGN above mean: %d/%d (%.1f%%)' % (N_y_tAGN_above, N_y_tAGN, 100.*N_y_tAGN_above/N_y_tAGN))
    ax_Hy.set_ylabel(ylabel)
    ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
    ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_Hy.set_ylim(yrange)
    ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
    ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
    tick_params = dict(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=False, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    ax_Hy.tick_params(**tick_params)
    ax_Hy_t.tick_params(**tick_params)
    ####################################
    sc = ax_sc.scatter(morph, y, **scatter_kwargs_EWmaxmin)
    mALLAGN = (elines['AGN_FLAG'] > 0)
    xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax_sc, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    # ax_sc.scatter(morph[mtIII], y[mtIII], **scatter_AGN_tIII_kwargs)
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
    cb.set_label(r'$\log {\rm W}_{{\rm H}\alpha}$ (\AA)', fontsize=args.fontsize+2)
    cb.locator = MaxNLocator(4)
    # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    ####################################
    if args.verbose > 0:
        print('# y #')
        ylim = ax_sc.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print('# N.y points < %.1f: %d' % (ylim[0], y_low.count()))
        if args.verbose > 1:
            for i in y_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.y points > %.1f: %d' % (ylim[1], y_upp.count()))
        if args.verbose > 1:
            for i in y_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
    return ax_Hx, ax_Hy, ax_sc


def plot_fig_histo_MZR(elines, args, x, y, ax):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    ### MZR ###
    mXnotnan = ~np.isnan(x)
    X = x.loc[mXnotnan].values
    iS = np.argsort(X)
    XS = X[iS]
    elines['modlogOHSF2017_t2'] = modlogOHSF2017_t2(elines['log_Mass_corr'])
    YS = (elines['modlogOHSF2017_t2'].loc[mXnotnan].values)[iS]
    mY = YS > 8.4
    mX = XS < 11.5
    ax.plot(XS[mX & mY], YS[mX & mY], 'k-')
    # ### best-fit ###
    # mnotnan = ~(np.isnan(x) | np.isnan(y)) & (x < 11.5) & (x > 8.3)
    # XFIT = x.loc[mnotnan].values
    # iSFIT = np.argsort(XFIT)
    # XFITS = XFIT[iSFIT]
    # YFIT = y.loc[mnotnan].values
    # YFITS = YFIT[iSFIT]
    # popt, pcov = curve_fit(f=modlogOHSF2017, xdata=XFITS, ydata=YFITS, p0=[8.8, 0.015, 11.5], bounds=[[8.54, 0.005, 11.499], [9, 0.022, 11.501]])
    # ax.plot(XFITS, modlogOHSF2017(XFITS, *popt), 'k--')
    # print('a:%.2f b:%.4f c:%.1f' % (popt[0], popt[1], popt[2]))
    ### Above ###
    m_y_tI_above = y.loc[mtI] > x.loc[mtI].apply(modlogOHSF2017_t2)
    m_y_tII_above = y.loc[mtII] > x.loc[mtII].apply(modlogOHSF2017_t2)
    # m_y_tIII_above = y.loc[mtIII] > x.loc[mtIII].apply(modlogOHSF2017_t2)
    m_y_tAGN_above = y.loc[mtAGN] > x.loc[mtAGN].apply(modlogOHSF2017_t2)
    print('AGN Type I:')
    print(elines.loc[m_y_tI_above.index[m_y_tI_above], ['log_Mass_corr', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']])
    print('AGN Type II:')
    print(elines.loc[m_y_tII_above.index[m_y_tII_above], ['log_Mass_corr', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']])
    # print(elines.loc[m_y_tIII_above.index[m_y_tIII_above], ['log_Mass_corr', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']])
    print('AGNs:')
    print(elines.loc[m_y_tAGN_above.index[m_y_tAGN_above], ['log_Mass_corr', 'OH_Re_fit_t2', 'modlogOHSF2017_t2']])
    N_y_tI = y[mtI].count()
    N_y_tII = y[mtII].count()
    # N_y_tIII = y[mtIII].count()
    N_y_tAGN = y[mtAGN].count()
    N_y_tI_above = m_y_tI_above.astype('int').sum()
    N_y_tII_above = m_y_tII_above.astype('int').sum()
    # N_y_tIII_above = m_y_tIII_above.astype('int').sum()
    N_y_tAGN_above = m_y_tAGN_above.astype('int').sum()
    print('# B.F. Type-I AGN above SF2017 curve: %d/%d (%.1f%%)' % (N_y_tI_above, N_y_tI, 100.*N_y_tI_above/N_y_tI))
    print('# B.F. Type-II AGN above SF2017 curve: %d/%d (%.1f%%)' % (N_y_tII_above, N_y_tII, 100.*N_y_tII_above/N_y_tII))
    # print('# 2 crit. AGN above SF2017 curve: %d/%d (%.1f%%)' % (N_y_tIII_above, N_y_tIII, 100.*N_y_tIII_above/N_y_tIII))
    print('# ALL AGN above mean: %d/%d (%.1f%%)' % (N_y_tAGN_above, N_y_tAGN, 100.*N_y_tAGN_above/N_y_tAGN))
    return ax


def plot_fig_histo_M_ZHMW(elines, args, x, y, ax, interval=None):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    if interval is None:
        interval = [9, 11.5, -0.9, 0.3]
    x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.3)
    _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(x.values, y.values, x_bins__r, interval)
    ax.plot(x_bincenter__r, y_mean, 'k-')
    ### above ###
    N_y_tI = y[mtI].count()
    N_y_tII = y[mtII].count()
    # N_y_tIII = y[mtIII].count()
    N_y_tAGN = y[mtAGN].count()
    N_y_tI_above = count_y_above_mean(x.loc[mtI].values, y.loc[mtI].values, y_mean, x_bins__r, interval=interval)
    N_y_tII_above = count_y_above_mean(x.loc[mtII].values, y.loc[mtII].values, y_mean, x_bins__r, interval=interval)
    # N_y_tIII_above = count_y_above_mean(x.loc[mtIII].values, y.loc[mtIII].values, y_mean, x_bins__r, interval=interval)
    N_y_tAGN_above = count_y_above_mean(x.loc[mtAGN].values, y.loc[mtAGN].values, y_mean, x_bins__r, interval=interval)
    print('# B.F. Type-I AGN above mean: %d/%d (%.1f%%)' % (N_y_tI_above, N_y_tI, 100.*N_y_tI_above/N_y_tI))
    print('# B.F. Type-II AGN above mean: %d/%d (%.1f%%)' % (N_y_tII_above, N_y_tII, 100.*N_y_tII_above/N_y_tII))
    # print('# 2 crit. AGN above mean: %d/%d (%.1f%%)' % (N_y_tIII_above, N_y_tIII, 100.*N_y_tIII_above/N_y_tIII))
    print('# ALL AGN above mean: %d/%d (%.1f%%)' % (N_y_tAGN_above, N_y_tAGN, 100.*N_y_tAGN_above/N_y_tAGN))
    return ax_sc


def plot_fig_histo_M_t(elines, args, x, y, ax, interval=None):
    WHa = elines['EW_Ha_ALL']
    WHa_Re =(-1. * elines['EW_Ha_Re'])
    WHa_ALL = elines['EW_Ha_ALL']
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    y_AGNs_mean = y.loc[mtAGN].mean()
    y_BF_AGNs_mean = y.loc[(mtAGN) & (elines['AGN_FLAG'] < 3)].mean()
    # ax_sc.axhline(y_AGNs_mean, xmin=0.9/(12.-8.), c='g', ls='--')
    ax_sc.axhline(y_AGNs_mean, c='g', ls='--')
    ax_sc.text(0.05, 0.85, '%.2f Gyr' % (10**(y_AGNs_mean - 9)), color='g', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    y_AGNs_tI_mean = y.loc[mtI].mean()
    y_AGNs_tII_mean = y.loc[mtII].mean()
    # y_AGNs_tIII_mean = y.loc[mtIII].mean()
    print('y_AGNs_mean: %.2f Gyr' % 10**(y_AGNs_mean - 9))
    print('y_BF_AGNs_mean: %.2f Gyr' % 10**(y_BF_AGNs_mean - 9))
    print('y_AGNs_tI_mean: %.2f Gyr' % 10**(y_AGNs_tI_mean - 9))
    print('y_AGNs_tII_mean: %.2f Gyr' % 10**(y_AGNs_tII_mean - 9))
    # print('y_AGNs_tIII_mean: %.2f Gyr' % 10**(y_AGNs_tIII_mean - 9))
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(WHa))
    hDIG = WHa <= args.EW_hDIG
    GV = (WHa > args.EW_hDIG) & (WHa <= args.EW_SF)
    SFc = WHa > args.EW_SF
    y_SF_mean = y.loc[m & SFc].mean()
    y_GV_mean = y.loc[m & GV].mean()
    y_hDIG_mean = y.loc[m & hDIG].mean()
    print('y_SF_mean: %.2f Gyr' % 10**(y_SF_mean - 9))
    print('y_GV_mean: %.2f Gyr' % 10**(y_GV_mean - 9))
    print('y_hDIG_mean: %.2f Gyr' % 10**(y_hDIG_mean - 9))
    ### MSFS ###
    ### SFG ###
    SFRHa = elines['lSFR']
    x_SF = x.loc[SFc]
    y_SF = y.loc[SFc]
    SFRHa_SF = SFRHa.loc[SFc]
    XS_SF, YS_SF, SFRHaS_SF = xyz_clean_sort_interval(x_SF.values, y_SF.values, SFRHa_SF.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, interval=interval)
    SFRHaS_SF_c__r, N_c__r, sel_c, SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_SF_c__r, 1)
    # print(10**(YS_SF[sel].mean()-9), 10**(YS_SF[sel_c].mean()-9))
    mean_t_SF = YS_SF[sel_c].mean()
    # ax_sc.axhline(mean_t_SF, xmin=0.9/(12.-8.), c='b', ls='--')
    ax_sc.axhline(mean_t_SF, c='b', ls='--')
    ax_sc.text(0.05, 0.77, '%.2f Gyr' % (10**(mean_t_SF - 9)), color='b', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    ### RG ###
    x_hDIG = x.loc[hDIG]
    y_hDIG = y.loc[hDIG]
    SFRHa_hDIG = SFRHa.loc[hDIG]
    XS_hDIG, YS_hDIG, SFRHaS_hDIG = xyz_clean_sort_interval(x_hDIG.values, y_hDIG.values, SFRHa_hDIG.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, interval=interval)
    SFRHaS_hDIG_c__r, N_c__r, sel_c, SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_hDIG_c__r, 1)
    # print(10**(YS_hDIG[sel].mean()-9), 10**(YS_hDIG[sel_c].mean()-9))
    mean_t_hDIG = YS_hDIG[sel_c].mean()
    # ax_sc.axhline(mean_t_hDIG, xmin=0.9/(12.-8.), c='r', ls='--')
    ax_sc.axhline(mean_t_hDIG, c='r', ls='--')
    ax_sc.text(0.05, 0.93, '%.2f Gyr' % (10**(mean_t_hDIG - 9)), color='r', fontsize=args.fontsize, va='center', transform=ax.transAxes)

    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(WHa_Re))
    print('### WHa_Re ###')
    hDIG = WHa_Re <= args.EW_hDIG
    GV = (WHa_Re > args.EW_hDIG) & (WHa_Re <= args.EW_SF)
    SFc = WHa_Re > args.EW_SF
    y_SF_mean = y.loc[m & SFc].mean()
    y_GV_mean = y.loc[m & GV].mean()
    y_hDIG_mean = y.loc[m & hDIG].mean()
    print('y_SF_mean: %.2f Gyr' % 10**(y_SF_mean - 9))
    print('y_GV_mean: %.2f Gyr' % 10**(y_GV_mean - 9))
    print('y_hDIG_mean: %.2f Gyr' % 10**(y_hDIG_mean - 9))
    ### MSFS ###
    ### SFG ###
    SFRHa = elines['lSFR']
    x_SF = x.loc[SFc]
    y_SF = y.loc[SFc]
    SFRHa_SF = SFRHa.loc[SFc]
    XS_SF, YS_SF, SFRHaS_SF = xyz_clean_sort_interval(x_SF.values, y_SF.values, SFRHa_SF.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, interval=interval)
    SFRHaS_SF_c__r, N_c__r, sel_c, SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_SF_c__r, 1)
    print(10**(YS_SF[sel].mean()-9), 10**(YS_SF[sel_c].mean()-9))
    mean_t_SF = YS_SF[sel_c].mean()
    # ax_sc.axhline(mean_t_SF, xmin=0.9/(12.-8.), c='b', ls='--')
    ### RG ###
    x_hDIG = x.loc[hDIG]
    y_hDIG = y.loc[hDIG]
    SFRHa_hDIG = SFRHa.loc[hDIG]
    XS_hDIG, YS_hDIG, SFRHaS_hDIG = xyz_clean_sort_interval(x_hDIG.values, y_hDIG.values, SFRHa_hDIG.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, interval=interval)
    SFRHaS_hDIG_c__r, N_c__r, sel_c, SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_hDIG_c__r, 1)
    print(10**(YS_hDIG[sel].mean()-9), 10**(YS_hDIG[sel_c].mean()-9))

    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(WHa_ALL))
    print('### WHa_ALL ###')
    hDIG = WHa_ALL <= args.EW_hDIG
    GV = (WHa_ALL > args.EW_hDIG) & (WHa_ALL <= args.EW_SF)
    SFc = WHa_ALL > args.EW_SF
    y_SF_mean = y.loc[m & SFc].mean()
    y_GV_mean = y.loc[m & GV].mean()
    y_hDIG_mean = y.loc[m & hDIG].mean()
    print('y_SF_mean: %.2f Gyr' % 10**(y_SF_mean - 9))
    print('y_GV_mean: %.2f Gyr' % 10**(y_GV_mean - 9))
    print('y_hDIG_mean: %.2f Gyr' % 10**(y_hDIG_mean - 9))
    ### MSFS ###
    ### SFG ###
    SFRHa = elines['lSFR']
    x_SF = x.loc[SFc]
    y_SF = y.loc[SFc]
    SFRHa_SF = SFRHa.loc[SFc]
    XS_SF, YS_SF, SFRHaS_SF = xyz_clean_sort_interval(x_SF.values, y_SF.values, SFRHa_SF.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, interval=interval)
    SFRHaS_SF_c__r, N_c__r, sel_c, SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_SF_c__r, 1)
    print(10**(YS_SF[sel].mean()-9), 10**(YS_SF[sel_c].mean()-9))
    ### RG ###
    x_hDIG = x.loc[hDIG]
    y_hDIG = y.loc[hDIG]
    SFRHa_hDIG = SFRHa.loc[hDIG]
    XS_hDIG, YS_hDIG, SFRHaS_hDIG = xyz_clean_sort_interval(x_hDIG.values, y_hDIG.values, SFRHa_hDIG.values)
    if interval is None:
        interval = [8.3, 11.8, 7.5, 10.5]
    x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, interval=interval)
    SFRHaS_hDIG_c__r, N_c__r, sel_c, SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_hDIG_c__r, 1)
    print(10**(YS_hDIG[sel].mean()-9), 10**(YS_hDIG[sel_c].mean()-9))
    return ax_sc


if __name__ == '__main__':
    args = parser_args()

    with open(args.input, 'rb') as f:
        pickled = pickle.load(f)

    elines = pickled['df']
    ###############################################################
    bug = pickled['bug']
    EW_SF = pickled['EW_SF']
    EW_AGN = pickled['EW_AGN']
    EW_hDIG = pickled['EW_hDIG']
    EW_strong = pickled['EW_strong']
    EW_verystrong = pickled['EW_verystrong']
    args.EW_SF = EW_SF
    args.EW_AGN = EW_AGN
    args.EW_hDIG = EW_hDIG
    args.EW_strong = EW_strong
    args.EW_verystrong = EW_verystrong
    sel_NIIHa = pickled['sel_NIIHa']
    sel_OIIIHb = pickled['sel_OIIIHb']
    sel_SIIHa = pickled['sel_SIIHa']
    sel_OIHa = pickled['sel_OIHa']
    sel_MS = pickled['sel_MS']
    sel_EW = pickled['sel_EW']
    sel_below_K01 = pickled['sel_below_K01']
    sel_below_K01_SII = pickled['sel_below_K01_SII']
    sel_below_K01_OI = pickled['sel_below_K01_OI']
    sel_below_K03 = pickled['sel_below_K03']
    sel_below_S06 = pickled['sel_below_S06']
    sel_below_CF10 = pickled['sel_below_CF10']
    sel_below_K06_SII = pickled['sel_below_K06_SII']
    sel_below_K06_OI = pickled['sel_below_K06_OI']
    sel_AGNLINER_NIIHa_OIIIHb = pickled['sel_AGNLINER_NIIHa_OIIIHb']
    sel_AGN_NIIHa_OIIIHb_K01_CF10 = pickled['sel_AGN_NIIHa_OIIIHb_K01_CF10']
    sel_AGN_SIIHa_OIIIHb_K01 = pickled['sel_AGN_SIIHa_OIIIHb_K01']
    sel_AGN_OIHa_OIIIHb_K01 = pickled['sel_AGN_OIHa_OIIIHb_K01']
    sel_AGN_SIIHa_OIIIHb_K01_K06 = pickled['sel_AGN_SIIHa_OIIIHb_K01_K06']
    sel_AGN_OIHa_OIIIHb_K01_K06 = pickled['sel_AGN_OIHa_OIIIHb_K01_K06']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01']
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01']
    sel_AGN_candidates = pickled['sel_AGN_candidates']
    sel_SAGN_candidates = pickled['sel_SAGN_candidates']
    sel_VSAGN_candidates = pickled['sel_VSAGN_candidates']
    sel_SF_NIIHa_OIIIHb_K01 = pickled['sel_SF_NIIHa_OIIIHb_K01']
    sel_SF_NIIHa_OIIIHb_K03 = pickled['sel_SF_NIIHa_OIIIHb_K03']
    sel_SF_NIIHa_OIIIHb_S06 = pickled['sel_SF_NIIHa_OIIIHb_S06']
    sel_pAGB = pickled['sel_pAGB']
    sel_SF_EW = pickled['sel_SF_EW']
    sel_AGNLINER_NIIHa_OIIIHb_MS = pickled['sel_AGNLINER_NIIHa_OIIIHb_MS']
    sel_AGN_NIIHa_OIIIHb_K01_CF10_MS = pickled['sel_AGN_NIIHa_OIIIHb_K01_CF10_MS']
    sel_AGN_SIIHa_OIIIHb_K01_MS = pickled['sel_AGN_SIIHa_OIIIHb_K01_MS']
    sel_AGN_OIHa_OIIIHb_K01_MS = pickled['sel_AGN_OIHa_OIIIHb_K01_MS']
    sel_AGN_SIIHa_OIIIHb_K01_K06_MS = pickled['sel_AGN_SIIHa_OIIIHb_K01_K06_MS']
    sel_AGN_OIHa_OIIIHb_K01_K06_MS = pickled['sel_AGN_OIHa_OIIIHb_K01_K06_MS']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS']
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS = pickled['sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS']
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS = pickled['sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS']
    sel_AGN_candidates_MS = pickled['sel_AGN_candidates_MS']
    sel_SAGN_candidates_MS = pickled['sel_SAGN_candidates_MS']
    sel_VSAGN_candidates_MS = pickled['sel_VSAGN_candidates_MS']
    sel_SF_NIIHa_OIIIHb_K01_MS = pickled['sel_SF_NIIHa_OIIIHb_K01_MS']
    sel_SF_NIIHa_OIIIHb_K03_MS = pickled['sel_SF_NIIHa_OIIIHb_K03_MS']
    sel_SF_NIIHa_OIIIHb_S06_MS = pickled['sel_SF_NIIHa_OIIIHb_S06_MS']
    sel_pAGB_MS = pickled['sel_pAGB_MS']
    sel_SF_EW_MS = pickled['sel_SF_EW_MS']
    ###############################################################
    log_NII_Ha_cen = elines['log_NII_Ha_cen']
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
    mtIII = elines['AGN_FLAG'] == 3
    N_AGN_tIII = mtIII.astype('int').sum()
    mtIV = elines['AGN_FLAG'] == 4
    N_AGN_tIV = mtIV.astype('int').sum()
    mBFAGN = mtI | mtII
    N_BFAGN = mBFAGN.astype('int').sum()
    mALLAGN = elines['AGN_FLAG'] > 0
    N_ALLAGN = mALLAGN.astype('int').sum()

    legend_elements = [
        Line2D([0], [0], marker=marker_AGN_tI, markeredgecolor=color_AGN_tI, label='Type-I (%d)' % N_AGN_tI, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
        Line2D([0], [0], marker=marker_AGN_tII, markeredgecolor=color_AGN_tII, label='Type-II (%d)' % N_AGN_tII, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
        Line2D([0], [0], marker=marker_AGN_tIII, alpha=alpha_AGN_tIII, markeredgecolor=color_AGN_tIII, label=r'by [NII]/H$\alpha$ and other (+%d)' % N_AGN_tIII, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
        Line2D([0], [0], marker=marker_AGN_tIV, alpha=alpha_AGN_tIV, markeredgecolor=color_AGN_tIV, label=r'by [NII]/H$\alpha$ (+%d)' % N_AGN_tIV, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
    ]

    ##########################
    ## BPT colored by EW_Ha ##
    ##########################
    print('\n##########################')
    print('## BPT colored by EW_Ha ##')
    print('##########################')
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
    print('##########################')
    print('## [NII]/Ha             ##')
    print('##########################')
    ax = ax0
    x = log_NII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs_EWmaxmin)
    ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
    ax.scatter(x.loc[mtII], y.loc[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y.loc[mtI], **scatter_AGN_tI_kwargs)
    ax.plot(L.x['K01'], L.y['K01'], 'k--')
    ax.plot(L.x['S06'], L.y['S06'], 'k-.')
    ax.plot(L.x['K03'], L.y['K03'], 'k-')
    ax.set_xlabel(r'$\log\ ({\rm [NII]}/{\rm H\alpha})$', fontsize=fs+4)
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN/LINER', 0.95, 0.87, fs+2, 'top', 'right', 'k')
    tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(**tick_params)
    ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
    ax.legend(handles=legend_elements, ncol=2, loc=2, frameon=False, fontsize='xx-small', borderpad=0, borderaxespad=0.75)
    ##########################
    # SII/Ha
    ##########################
    print('##########################')
    print('## [SII]/Ha             ##')
    print('##########################')
    ax = ax1
    x = log_SII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs_EWmaxmin)
    ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
    ax.scatter(x.loc[mtII], y.loc[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y.loc[mtI], **scatter_AGN_tI_kwargs)
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
    # ax.legend(handles=legend_elements, loc=2, frameon=False, fontsize='x-small', borderpad=0, borderaxespad=0.5)  # markerfirst=False)
    ##########################
    # OI/Ha
    ##########################
    print('##########################')
    print('## [OI]/Ha              ##')
    print('##########################')
    ax = ax2
    x = log_OI_Ha_cen
    extent = [-3, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs_EWmaxmin)
    ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
    ax.scatter(x.loc[mtII], y.loc[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y.loc[mtI], **scatter_AGN_tI_kwargs)
    ax.set_xlabel(r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', fontsize=fs+4)
    cb_ax = f.add_axes([right, bottom, 0.02, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'$\log {\rm W}_{{\rm H}\alpha}$ (\AA)', fontsize=fs+4)
    cb_ax.tick_params(direction='in')
    cb.locator = MaxNLocator(4)
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
    # ax.legend(handles=legend_elements, loc=4, ncol=2, frameon=False, fontsize='xx-small', borderpad=0, borderaxespad=0.75)  # markerfirst=False)
    # ax.legend(handles=legend_elements, markerfirst=False, loc=4, frameon=False, fontsize='x-small', borderpad=0, borderaxespad=0.5)
    ##########################
    f.text(0.01, 0.5, r'$\log\ ({\rm [OIII]}/{\rm H\beta})$', va='center', rotation='vertical', fontsize=fs+4)
    f.savefig('%s/fig_BPT.%s' % (args.figs_dir, args.img_suffix), dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('##########################')
    ##########################

    ##################################
    ## WHAN colored by [OIII]/Hbeta ##
    ##################################
    print('\n##################################')
    print('## WHAN colored by [OIII]/Hbeta ##')
    print('##################################')
    x = log_NII_Ha_cen
    y = EW_Ha_cen
    z = log_OIII_Hb_cen
    extent = [-2, 1.5, -1, 3]
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_WHAN(args, x.values, y.values, f=f, ax=ax, extent=extent,
                      z=z.values, z_label=r'$\log\ ({\rm [OIII]}/{\rm H\beta})$',
                      cmap='viridis_r', mask=None,
                      vmax=1, vmin=-1)
    y = EW_Ha_cen.apply(np.log10)
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    # sns.kdeplot(x.loc[mtI], y.loc[mtI], ax=plt.gca(), color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    # ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    # ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
    ax.scatter(x.loc[mtII], y.loc[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y.loc[mtI], **scatter_AGN_tI_kwargs)
    ####################################
    if args.verbose > 0:
        print('# x #')
        xlim = ax.get_xlim()
        x_low = x.loc[x < xlim[0]]
        x_upp = x.loc[x > xlim[1]]
        print('# N.x points < %.1f: %d' % (xlim[0], x_low.count()))
        if args.verbose > 1:
            for i in x_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.x points > %.1f: %d' % (xlim[1], x_upp.count()))
        if args.verbose > 1:
            for i in x_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
        print('# y #')
        ylim = ax.get_ylim()
        y_low = y.loc[y < ylim[0]]
        y_upp = y.loc[y > ylim[1]]
        print('# N.y points < %.1f: %d' % (ylim[0], y_low.count()))
        if args.verbose > 1:
            for i in y_low.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('# N.y points > %.1f: %d' % (ylim[1], y_upp.count()))
        if args.verbose > 1:
            for i in y_upp.index:
                print('#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG']))
        print('#####')
    ####################################
    output_name='%s/fig_WHAN.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('##################################')
    ##################################


    ###########################
    ## SFMS colored by EW_Ha ##
    ###########################
    print('\n###########################')
    print('## SFMS colored by EW_Ha ##')
    print('###########################')
    x = elines['log_Mass_corr']
    y = elines['lSFR']
    z = EW_Ha_cen.apply(np.log10)
    xlabel = r'$\log ({\rm M}_\star/{\rm M}_{\odot})$'
    extent = [7.5, 12, -4.5, 2.5]
    n_bins_min_x = 5
    n_bins_maj_y = 4
    n_bins_min_y = 2
    prune_x = None
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                      ylabel=r'$\log ({\rm SFR_{{\rm H}}\alpha}/{\rm M}_{\odot}/{\rm yr})$',
                      xlabel=xlabel, extent=extent,
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      f=f, ax=ax)
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    WHa = EW_Ha_cen
    WHa_Re = (-1. * elines['EW_Ha_Re'])
    WHa_ALL = elines['EW_Ha_ALL']
    hDIG = sel_EW & (WHa <= args.EW_hDIG)
    SFc = sel_EW & (WHa > args.EW_SF)
    hDIG_Re = sel_EW & (WHa_Re <= args.EW_hDIG)
    SFc_Re = sel_EW & (WHa_Re > args.EW_SF)
    hDIG_ALL = sel_EW & (WHa_ALL <= args.EW_hDIG)
    SFc_ALL = sel_EW & (WHa_ALL > args.EW_SF)
    interval = [8.3, 11.8, 7.5, 10.5]
    dict_masks = dict(hDIG=hDIG, hDIG_Re=hDIG_Re, hDIG_ALL=hDIG_ALL, SFc=SFc, SFc_Re=SFc_Re, SFc_ALL=SFc_ALL)
    for k, v in dict_masks.items():
        print('{}:'.format(k))
        X = x.loc[v]
        Y = y.loc[v]
        p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel = linear_regression_mean(X, Y, interval=interval, step=0.1, clip=2)
        if k == 'SFc':
            p_SFc = p
            p_SFc_c = pc
            ax.plot(interval[0:2], np.polyval(p_SFc_c, interval[0:2]), c='k', label='SFG')
            ax.text(x_bins_center__r[0], np.polyval(p_SFc_c, x_bins_center__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
        if k == 'hDIG':
            p_hDIG = p
            p_hDIG_c = pc
            ax.plot(interval[0:2], np.polyval(p_hDIG_c, interval[0:2]), c='k', ls='--', label='RG')
            ax.text(x_bins_center__r[0], np.polyval(p_hDIG_c, x_bins_center__r[0]), 'RG', color='k', fontsize=args.fontsize, va='center', ha='right')
    ###########################
    N_AGN_tI_under_SF = ((y[mtI] - np.polyval(p_SFc_c, x[mtI])) <= 0).astype('int').sum()
    N_AGN_tII_under_SF = ((y[mtII] - np.polyval(p_SFc_c, x[mtII])) <= 0).astype('int').sum()
    N_BFAGN_under_SF = ((y[mBFAGN] - np.polyval(p_SFc_c, x[mBFAGN])) <= 0).astype('int').sum()
    N_ALLAGN_under_SF = ((y[mALLAGN] - np.polyval(p_SFc_c, x[mALLAGN])) <= 0).astype('int').sum()
    print('# B.F. Type-I AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tI_under_SF, N_AGN_tI, 100.*N_AGN_tI_under_SF/N_AGN_tI))
    print('# B.F. Type-II AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tII_under_SF, N_AGN_tII, 100.*N_AGN_tII_under_SF/N_AGN_tII))
    print('# B.F. AGN under SFc curve: %d/%d (%.1f%%)' % (N_BFAGN_under_SF, N_BFAGN, 100.*N_BFAGN_under_SF/N_BFAGN))
    print('# ALL AGN under SFc curve: %d/%d (%.1f%%)' % (N_ALLAGN_under_SF, N_ALLAGN, 100.*N_ALLAGN_under_SF/N_ALLAGN))
    ###########################
    output_name = '%s/fig_SFMS.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('###########################')
    ###########################
    print('##############')
    print('## (NO CEN)) ##')
    print('##############')
    x = elines['log_Mass_corr_NC']
    y = elines['lSFR_NC']
    z = EW_Ha_cen.apply(np.log10)
    xlabel = r'$\log ({\rm M}_\star/{\rm M}_{\odot})$'
    extent = [7.5, 12, -4.5, 2.5]
    n_bins_min_x = 5
    n_bins_maj_y = 4
    n_bins_min_y = 2
    prune_x = None
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                      ylabel=r'$\log ({\rm SFR_{{\rm H}}\alpha}/{\rm M}_{\odot}/{\rm yr})$',
                      xlabel=xlabel, extent=extent,
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      f=f, ax=ax)
    plot_text_ax(ax, 'NOCEN', 0.04, 0.95, fs+2, 'top', 'left', 'k')
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    WHa = EW_Ha_cen
    WHa_Re =(-1. * elines['EW_Ha_Re'])
    WHa_ALL = elines['EW_Ha_ALL']
    hDIG = sel_EW & (WHa <= args.EW_hDIG)
    SFc = sel_EW & (WHa > args.EW_SF)
    hDIG_Re = sel_EW & (WHa_Re <= args.EW_hDIG)
    SFc_Re = sel_EW & (WHa_Re > args.EW_SF)
    hDIG_ALL = sel_EW & (WHa_ALL <= args.EW_hDIG)
    SFc_ALL = sel_EW & (WHa_ALL > args.EW_SF)
    interval = [8.3, 11.8, 7.5, 10.5]
    dict_masks = dict(hDIG=hDIG, hDIG_Re=hDIG_Re, hDIG_ALL=hDIG_ALL, SFc=SFc, SFc_Re=SFc_Re, SFc_ALL=SFc_ALL)
    for k, v in dict_masks.items():
        print('{}:'.format(k))
        X = x.loc[v]
        Y = y.loc[v]
        p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel = linear_regression_mean(X, Y, interval=interval, step=0.1, clip=2)
        if k == 'SFc':
            p_SFc = p
            p_SFc_c = pc
            ax.plot(interval[0:2], np.polyval(p_SFc_c, interval[0:2]), c='k', label='SFG')
            ax.text(x_bins_center__r[0], np.polyval(p_SFc_c, x_bins_center__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
        if k == 'hDIG':
            p_hDIG = p
            p_hDIG_c = pc
            ax.plot(interval[0:2], np.polyval(p_hDIG_c, interval[0:2]), c='k', ls='--', label='RG')
            ax.text(x_bins_center__r[0], np.polyval(p_hDIG_c, x_bins_center__r[0]), 'RG', color='k', fontsize=args.fontsize, va='center', ha='right')
    ###########################
    N_AGN_tI_under_SF = ((y[mtI] - np.polyval(p_SFc, x[mtI])) <= 0).astype('int').sum()
    N_AGN_tII_under_SF = ((y[mtII] - np.polyval(p_SFc, x[mtII])) <= 0).astype('int').sum()
    N_BFAGN_under_SF = ((y[mBFAGN] - np.polyval(p_SFc, x[mBFAGN])) <= 0).astype('int').sum()
    N_ALLAGN_under_SF = ((y[mALLAGN] - np.polyval(p_SFc, x[mALLAGN])) <= 0).astype('int').sum()
    print('# B.F. Type-I AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tI_under_SF, N_AGN_tI, 100.*N_AGN_tI_under_SF/N_AGN_tI))
    print('# B.F. Type-II AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tII_under_SF, N_AGN_tII, 100.*N_AGN_tII_under_SF/N_AGN_tII))
    print('# B.F. AGN under SFc curve: %d/%d (%.1f%%)' % (N_BFAGN_under_SF, N_BFAGN, 100.*N_BFAGN_under_SF/N_BFAGN))
    print('# ALL AGN under SFc curve: %d/%d (%.1f%%)' % (N_ALLAGN_under_SF, N_ALLAGN, 100.*N_ALLAGN_under_SF/N_ALLAGN))
    ###########################
    output_name = '%s/fig_SFMS_NC.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('###########################')
    ################################

    ########################################
    ## SFMS SSP (10Myr) colored by EW_Ha ##
    ########################################
    print('\n###############################')
    print('## SFMS SSP colored by EW_Ha ##')
    print('###############################')
    x = elines['log_Mass_corr']
    y = elines['log_SFR_ssp_10Myr']
    z = EW_Ha_cen.apply(np.log10)
    xlabel = r'$\log ({\rm M}_\star/{\rm M}_{\odot})$'
    extent = [7.5, 12, -4.5, 2.5]
    n_bins_min_x = 5
    n_bins_maj_y = 4
    n_bins_min_y = 5
    prune_x = None
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                      ylabel=r'$\log ({\rm SFR_{\rm SSP}}/{\rm M}_{\odot}/{\rm yr})$',
                      xlabel=xlabel, extent=extent,
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      f=f, ax=ax)
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    WHa = EW_Ha_cen
    WHa_Re =(-1. * elines['EW_Ha_Re'])
    WHa_ALL = elines['EW_Ha_ALL']
    hDIG = sel_EW & (WHa <= args.EW_hDIG)
    SFc = sel_EW & (WHa > args.EW_SF)
    hDIG_Re = sel_EW & (WHa_Re <= args.EW_hDIG)
    SFc_Re = sel_EW & (WHa_Re > args.EW_SF)
    hDIG_ALL = sel_EW & (WHa_ALL <= args.EW_hDIG)
    SFc_ALL = sel_EW & (WHa_ALL > args.EW_SF)
    interval = [8.3, 11.8, 7.5, 10.5]
    dict_masks = dict(hDIG=hDIG, hDIG_Re=hDIG_Re, hDIG_ALL=hDIG_ALL, SFc=SFc, SFc_Re=SFc_Re, SFc_ALL=SFc_ALL)
    for k, v in dict_masks.items():
        print('{}:'.format(k))
        X = x.loc[v]
        Y = y.loc[v]
        p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel = linear_regression_mean(X, Y, interval=interval, step=0.1, clip=2)
        if k == 'SFc':
            p_SFc = p
            p_SFc_c = pc
            ax.plot(interval[0:2], np.polyval(p_SFc_c, interval[0:2]), c='k', label='SFG')
            ax.text(x_bins_center__r[0], np.polyval(p_SFc_c, x_bins_center__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
        if k == 'hDIG':
            p_hDIG = p
            p_hDIG_c = pc
            ax.plot(interval[0:2], np.polyval(p_hDIG_c, interval[0:2]), c='k', ls='--', label='RG')
            ax.text(x_bins_center__r[0], np.polyval(p_hDIG_c, x_bins_center__r[0]), 'RG', color='k', fontsize=args.fontsize, va='center', ha='right')
    ########################################
    N_AGN_tI_under_SF = ((y[mtI] - np.polyval(p_SFc, x[mtI])) <= 0).astype('int').sum()
    N_AGN_tII_under_SF = ((y[mtII] - np.polyval(p_SFc, x[mtII])) <= 0).astype('int').sum()
    N_BFAGN_under_SF = ((y[mBFAGN] - np.polyval(p_SFc, x[mBFAGN])) <= 0).astype('int').sum()
    N_ALLAGN_under_SF = ((y[mALLAGN] - np.polyval(p_SFc, x[mALLAGN])) <= 0).astype('int').sum()
    print('# B.F. Type-I AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tI_under_SF, N_AGN_tI, 100.*N_AGN_tI_under_SF/N_AGN_tI))
    print('# B.F. Type-II AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tII_under_SF, N_AGN_tII, 100.*N_AGN_tII_under_SF/N_AGN_tII))
    print('# B.F. AGN under SFc curve: %d/%d (%.1f%%)' % (N_BFAGN_under_SF, N_BFAGN, 100.*N_BFAGN_under_SF/N_BFAGN))
    print('# ALL AGN under SFc curve: %d/%d (%.1f%%)' % (N_ALLAGN_under_SF, N_ALLAGN, 100.*N_ALLAGN_under_SF/N_ALLAGN))
    ########################################
    output_name = '%s/fig_SFMS_ssp10.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('###############################')
    ########################################

    ########################################
    ## SFMS SSP (100Myr) colored by EW_Ha ##
    ########################################
    print('\n###############################')
    print('## SFMS SSP colored by EW_Ha ##')
    print('###############################')
    x = elines['log_Mass_corr']
    y = elines['log_SFR_ssp_100Myr']
    z = EW_Ha_cen.apply(np.log10)
    xlabel = r'$\log ({\rm M}_\star/{\rm M}_{\odot})$'
    extent = [7.5, 12, -4.5, 2.5]
    n_bins_min_x = 5
    n_bins_maj_y = 4
    n_bins_min_y = 5
    prune_x = None
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                      ylabel=r'$\log ({\rm SFR_{\rm SSP}}/{\rm M}_{\odot}/{\rm yr})$',
                      xlabel=xlabel, extent=extent,
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      f=f, ax=ax)
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
    # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
    WHa = EW_Ha_cen
    WHa_Re =(-1. * elines['EW_Ha_Re'])
    WHa_ALL = elines['EW_Ha_ALL']
    hDIG = sel_EW & (WHa <= args.EW_hDIG)
    SFc = sel_EW & (WHa > args.EW_SF)
    hDIG_Re = sel_EW & (WHa_Re <= args.EW_hDIG)
    SFc_Re = sel_EW & (WHa_Re > args.EW_SF)
    hDIG_ALL = sel_EW & (WHa_ALL <= args.EW_hDIG)
    SFc_ALL = sel_EW & (WHa_ALL > args.EW_SF)
    interval = [8.3, 11.8, 7.5, 10.5]
    dict_masks = dict(hDIG=hDIG, hDIG_Re=hDIG_Re, hDIG_ALL=hDIG_ALL, SFc=SFc, SFc_Re=SFc_Re, SFc_ALL=SFc_ALL)
    for k, v in dict_masks.items():
        print('{}:'.format(k))
        X = x.loc[v]
        Y = y.loc[v]
        p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel = linear_regression_mean(X, Y, interval=interval, step=0.1, clip=2)
        if k == 'SFc':
            p_SFc = p
            p_SFc_c = pc
            ax.plot(interval[0:2], np.polyval(p_SFc_c, interval[0:2]), c='k', label='SFG')
            ax.text(x_bins_center__r[0], np.polyval(p_SFc_c, x_bins_center__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
        if k == 'hDIG':
            p_hDIG = p
            p_hDIG_c = pc
            ax.plot(interval[0:2], np.polyval(p_hDIG_c, interval[0:2]), c='k', ls='--', label='RG')
            ax.text(x_bins_center__r[0], np.polyval(p_hDIG_c, x_bins_center__r[0]), 'RG', color='k', fontsize=args.fontsize, va='center', ha='right')
    ########################################
    N_AGN_tI_under_SF = ((y[mtI] - np.polyval(p_SFc, x[mtI])) <= 0).astype('int').sum()
    N_AGN_tII_under_SF = ((y[mtII] - np.polyval(p_SFc, x[mtII])) <= 0).astype('int').sum()
    N_BFAGN_under_SF = ((y[mBFAGN] - np.polyval(p_SFc, x[mBFAGN])) <= 0).astype('int').sum()
    N_ALLAGN_under_SF = ((y[mALLAGN] - np.polyval(p_SFc, x[mALLAGN])) <= 0).astype('int').sum()
    print('# B.F. Type-I AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tI_under_SF, N_AGN_tI, 100.*N_AGN_tI_under_SF/N_AGN_tI))
    print('# B.F. Type-II AGN under SFc curve: %d/%d (%.1f%%)' % (N_AGN_tII_under_SF, N_AGN_tII, 100.*N_AGN_tII_under_SF/N_AGN_tII))
    print('# B.F. AGN under SFc curve: %d/%d (%.1f%%)' % (N_BFAGN_under_SF, N_BFAGN, 100.*N_BFAGN_under_SF/N_BFAGN))
    print('# ALL AGN under SFc curve: %d/%d (%.1f%%)' % (N_ALLAGN_under_SF, N_ALLAGN, 100.*N_ALLAGN_under_SF/N_ALLAGN))
    ########################################
    output_name = '%s/fig_SFMS_ssp100.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('###############################')
    ########################################

    ###############################
    ## Mgas-SFR colored by EW_Ha ##
    ###############################
    print('\n###############################')
    print('## Mgas-SFR colored by EW_Ha ##')
    print('###############################')
    x = elines['log_Mass_gas_Av_gas_rad']
    xlabel = r'$\log ({\rm M}_{\rm gas,A_V}/{\rm M}_{\odot})$'
    extent = [5, 11, -4.5, 2.5]
    n_bins_maj_x = 6
    n_bins_min_x = 5
    n_bins_maj_y = 4
    n_bins_min_y = 4
    prune_x = None
    plot_colored_by_z(elines=elines, args=args, x=x, y=elines['lSFR'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                      ylabel=r'$\log ({\rm SFR}/{\rm M}_{\odot}/{\rm yr})$',
                      xlabel=xlabel, extent=extent,
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      output_name='%s/fig_Mgas_SFR.%s' % (args.figs_dir, args.img_suffix))
    ###############################

    #############################
    ## M-Mgas colored by EW_Ha ##
    #############################
    print('\n#############################')
    print('## M-Mgas colored by EW_Ha ##')
    print('#############################')
    n_bins_min_x = 5
    n_bins_maj_y = 6
    n_bins_min_y = 5
    plot_colored_by_z(elines=elines, args=args, x=elines['log_Mass_corr'], y=elines['log_Mass_gas_Av_gas_rad'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                      xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                      ylabel=r'$\log ({\rm M}_{\rm gas,A_V}/{\rm M}_{\odot})$',
                      extent=[8, 12.5, 5, 11],
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      output_name='%s/fig_M_Mgas.%s' % (args.figs_dir, args.img_suffix))
    print('#############################')
    #############################

    ##########################
    ## M-C colored by EW_Ha ##
    ##########################
    print('\n##########################')
    print('## M-C colored by EW_Ha ##')
    print('##########################')
    n_bins_min_x = 5
    n_bins_maj_y = 6
    n_bins_min_y = 2
    plot_colored_by_z(elines=elines, args=args, x=elines['log_Mass_corr'], y=elines['C'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                      xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                      ylabel=r'${\rm R}90/{\rm R}50$',
                      extent=[8, 12.5, 0.5, 5.5],
                      n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                      n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                      output_name='%s/fig_M_C.%s' % (args.figs_dir, args.img_suffix))
    print('##########################')
    ##########################

    #############################
    ## sSFR-C colored by EW_Ha ##
    #############################
    print('\n#############################')
    print('## sSFR-C colored by EW_Ha ##')
    print('#############################')
    x = elines['sSFR']
    y = elines['C']
    z = EW_Ha_cen.apply(np.log10)
    n_bins_min_x = 5
    n_bins_maj_y = 6
    n_bins_min_y = 2
    output_name = '%s/fig_sSFR_C.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
                              xlabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                              ylabel=r'${\rm R}90/{\rm R}50$',
                              extent=[-13.5, -8.5, 0.5, 5.5], markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_min_x=n_bins_min_x, prune_x=prune_x)
    N_GV = ((x <= -10.8) & (x > -11.8)).astype('int').sum()
    N_AGN_tI_GV = ((x[mtI] <= -10.8) & (x[mtI] > -11.8)).astype('int').sum()
    N_AGN_tII_GV = ((x[mtII] <= -10.8) & (x[mtII] > -11.8)).astype('int').sum()
    N_BFAGN_GV = ((x[mBFAGN] <= -10.8) & (x[mBFAGN] > -11.8)).astype('int').sum()
    N_ALLAGN_GV = ((x[mALLAGN] <= -10.8) & (x[mALLAGN] > -11.8)).astype('int').sum()
    print('# B.F. Type-I AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tI_GV, N_GV, N_AGN_tI, 100.*N_AGN_tI_GV/N_GV, 100.*N_AGN_tI_GV/N_AGN_tI))
    print('# B.F. Type-II AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tII_GV, N_GV, N_AGN_tII, 100.*N_AGN_tII_GV/N_GV, 100.*N_AGN_tII_GV/N_AGN_tII))
    print('# B.F. AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_BFAGN_GV, N_GV, N_BFAGN, 100.*N_BFAGN_GV/N_GV, 100.*N_BFAGN_GV/N_BFAGN))
    print('# ALL AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_ALLAGN_GV, N_GV, N_ALLAGN, 100.*N_ALLAGN_GV/N_GV, 100.*N_ALLAGN_GV/N_ALLAGN))
    ax.axvline(x=-11.8, c='k', ls='--')
    ax.axvline(x=-10.8, c='k', ls='--')

    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('#############################')
    ##########################

    #############################
    ## M-sSFR colored by EW_Ha ##
    #############################
    print('\n#############################')
    print('## M-sSFR colored by EW_Ha ##')
    print('#############################')
    x = elines['log_Mass_corr']
    y = elines['sSFR']
    z = EW_Ha_cen.apply(np.log10)
    n_bins_min_x = 5
    n_bins_maj_y = 5
    n_bins_min_y = 2
    output_name = '%s/fig_M_sSFR.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
                              ylabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                              xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                              extent=[8, 12.5, -13.5, -8.5], markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_min_x=n_bins_min_x, prune_x=prune_x)
    ax.axhline(y=-11.8, c='k', ls='--')
    ax.axhline(y=-10.8, c='k', ls='--')
    N_GV = ((y <= -10.8) & (y > -11.8)).astype('int').sum()
    N_AGN_tI_GV = ((y[mtI] <= -10.8) & (y[mtI] > -11.8)).astype('int').sum()
    N_AGN_tII_GV = ((y[mtII] <= -10.8) & (y[mtII] > -11.8)).astype('int').sum()
    N_BFAGN_GV = ((y[mBFAGN] <= -10.8) & (y[mBFAGN] > -11.8)).astype('int').sum()
    N_ALLAGN_GV = ((y[mALLAGN] <= -10.8) & (y[mALLAGN] > -11.8)).astype('int').sum()
    print('# B.F. Type-I AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tI_GV, N_GV, N_AGN_tI, 100.*N_AGN_tI_GV/N_GV, 100.*N_AGN_tI_GV/N_AGN_tI))
    print('# B.F. Type-II AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tII_GV, N_GV, N_AGN_tII, 100.*N_AGN_tII_GV/N_GV, 100.*N_AGN_tII_GV/N_AGN_tII))
    print('# B.F. AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_BFAGN_GV, N_GV, N_BFAGN, 100.*N_BFAGN_GV/N_GV, 100.*N_BFAGN_GV/N_BFAGN))
    print('# ALL AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_ALLAGN_GV, N_GV, N_ALLAGN, 100.*N_ALLAGN_GV/N_GV, 100.*N_ALLAGN_GV/N_ALLAGN))
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('#############################')
    ##########################

    ################################
    ## X Y histo colored by EW_Ha ##
    ################################
    print('\n################################')
    print('## X Y histo colored by EW_Ha ##')
    print('################################')
    # print(elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean'])
    # print(np.log10(np.abs(elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean'])))
    m_redshift = (elines['z_stars'] > 1e-6) & (elines['z_stars'] < 0.2)
    EW_Ha_cen_zcut = elines.loc[m_redshift, 'EW_Ha_cen_mean'].apply(np.abs)
    plot_histo_xy_dict = {
        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_ui_CUBES': [
            elines.loc[m_redshift, 'Mabs_i'], r'${\rm M}_{\rm i}$ (mag)', 5, 2, None,
            elines.loc[m_redshift, 'u_i'], r'u-i (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 3.5],
            m_redshift
        ],
        ##################################
        #########################################
        ## CMD (CUBES) NO CEN colored by EW_Ha ##
        #########################################
        'fig_histo_CMD_ui_CUBES_NC': [
            elines.loc[m_redshift, 'Mabs_i_NC'], r'${\rm M}_{\rm i}$ (mag)', 5, 2, None,
            elines.loc[m_redshift, 'u_i_NC'], r'u-i (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 3.5],
            m_redshift
        ],
        #########################################
        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_ur_CUBES': [
            elines.loc[m_redshift, 'Mabs_r'], r'${\rm M}_{\rm r}$ (mag)', 5, 2, None,
            elines.loc[m_redshift, 'u_r'], r'u-r (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 3.5],
            m_redshift
        ],
        ##################################
        #########################################
        ## CMD (CUBES) NO CEN colored by EW_Ha ##
        #########################################
        'fig_histo_CMD_ur_CUBES_NC': [
            elines.loc[m_redshift, 'Mabs_r_NC'], r'${\rm M}_{\rm r}$ (mag)', 5, 2, None,
            elines.loc[m_redshift, 'u_r_NC'], r'u-r (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 3.5],
            m_redshift
        ],
        #########################################
        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_BR_CUBES': [
            elines.loc[m_redshift, 'Mabs_R'], r'${\rm M}_{\rm R}$ (mag)', 5, 5, None,
            elines.loc[m_redshift, 'B_R'], r'B-R (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 1.5],
            m_redshift
        ],
        ##################################
        #########################################
        ## CMD (CUBES) NO CEN colored by EW_Ha ##
        #########################################
        'fig_histo_CMD_BR_CUBES_NC': [
            # elines.loc[m_redshift, 'Mabs_R_NC'], r'${\rm M}_{\rm R}$ (mag)${}_{\rm NO CEN}$', 5, 5, None,
            # elines.loc[m_redshift, 'B_R_NC'], r'B-R (mag)${}_{\rm NO CEN}$', 3, 5, None,
            elines.loc[m_redshift, 'Mabs_R_NC'], r'${\rm M}_{\rm R}$ (mag)', 5, 5, None,
            elines.loc[m_redshift, 'B_R_NC'], r'B-R (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0, 1.5],
            m_redshift
        ],
        #########################################
        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_BV_CUBES': [
            elines.loc[m_redshift, 'Mabs_V'], r'${\rm M}_{\rm V}$ (mag)', 5, 5, None,
            elines.loc[m_redshift, 'B_V'], r'B-V (mag)', 2, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0., 1.0],
            m_redshift
        ],
        ##################################
        #########################################
        ## CMD (CUBES) NO CEN colored by EW_Ha ##
        #########################################
        'fig_histo_CMD_BV_CUBES_NC': [
            # elines.loc[m_redshift, 'Mabs_V_NC'], r'${\rm M}_{\rm V}$ (mag)${}_{\rm NO CEN}$', 5, 5, None,
            # elines.loc[m_redshift, 'B_V_NC'], r'B-V (mag)${}_{\rm NO CEN}$', 3, 5, None,
            elines.loc[m_redshift, 'Mabs_V_NC'], r'${\rm M}_{\rm V}$ (mag)', 5, 5, None,
            elines.loc[m_redshift, 'B_V_NC'], r'B-V (mag)', 2, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-24, -15, 0., 1.0],
            m_redshift
        ],
        #########################################
        ##################################
        ## sSFR vs u-i colored by EW_Ha ##
        ##################################
        'fig_histo_sSFR_ui': [
            elines.loc[m_redshift, 'lSFR'] - elines.loc[m_redshift, 'log_Mass_corr'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
            elines.loc[m_redshift, 'u_i'], r'u-i (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-13.5, -8.5, 0, 3.5],
            m_redshift
        ],
        ##################################
        ##################################
        ## sSFR vs u-r colored by EW_Ha ##
        ##################################
        'fig_histo_sSFR_ur': [
            elines.loc[m_redshift, 'sSFR'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
            elines.loc[m_redshift, 'u_r'], r'u-r (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-13.5, -8.5, 0, 3.5],
            m_redshift
        ],
        ##################################
        ##################################
        ## sSFR vs B-R colored by EW_Ha ##
        ##################################
        'fig_histo_sSFR_BR': [
            elines.loc[m_redshift, 'sSFR'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
            elines.loc[m_redshift, 'B_R'], r'B-R (mag)', 3, 5, None,
            EW_Ha_cen_zcut.apply(np.log10), [-13.5, -8.5, 0, 1.5],
            m_redshift
        ],
        ##################################
        ##################################
        ## sSFR vs B-V colored by EW_Ha ##
        ##################################
        'fig_histo_sSFR_BV': [
        elines.loc[m_redshift, 'sSFR'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
        elines.loc[m_redshift, 'B_V'], r'B-V (mag)', 3, 5, None,
        EW_Ha_cen_zcut.apply(np.log10), [-13.5, -8.5, -0.5, 1.0],
        m_redshift
        ],
        ##################################
        ###########################
        ## SFMS colored by EW_Ha ##
        ###########################
        'fig_histo_SFMS': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['lSFR'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', 4, 2, None,
            EW_Ha_cen.apply(np.log10), [8, 12, -4.5, 2.5]
        ],
        ###########################
        ####################################
        ## SFMS colored by EW_Ha (NO CEN) ##
        ####################################
        'fig_histo_SFMS_NC': [
            elines['log_Mass_corr_NC'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['lSFR_NC'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', 4, 2, None,
            EW_Ha_cen.apply(np.log10), [8, 12, -4.5, 2.5]
        ],
        ####################################
        ##########################
        ## M-C colored by EW_Ha ##
        ##########################
        'fig_histo_M_C': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['C'], r'${\rm R}90/{\rm R}50$', 6, 2, None,
            EW_Ha_cen.apply(np.log10), [8, 12, 0.5, 5.5]
        ],
        ##########################
        #############################
        ## sSFR-C colored by EW_Ha ##
        #############################
        'fig_histo_sSFR_C': [
            elines['lSFR'] - elines['log_Mass_corr'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
            elines['C'], r'${\rm R}90/{\rm R}50$', 6, 2, None,
            EW_Ha_cen.apply(np.log10), [-13.5, -8.5, 0.5, 5.5]
        ],
        #############################
        #############################
        ## M-sSFR colored by EW_Ha ##
        #############################
        'fig_histo_M_sSFR': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['lSFR'] - elines['log_Mass_corr'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None,
            EW_Ha_cen.apply(np.log10), [8, 12, -13.5, -8.5]
        ],
        #############################
        ###############################
        ## M-ZHLWRe colored by EW_Ha ##
        ###############################
        'fig_histo_M_ZHLW': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['ZH_LW_Re_fit'], r'[Z/H] LW', 3, 2, None,
            EW_Ha_cen.apply(np.log10), [8, 12, -0.7, 0.3]
            # elines['ZH_LW_Re_fit'] - elines['alpha_ZH_LW_Re_fit'], r'[Z/H] LW', 3, 2, None,
            # EW_Ha_cen.apply(np.log10), [8, 12, -0.9, 0.3]
        ],
        ###############################
        ###############################
        ## M-ZHMWRe colored by EW_Ha ##
        ###############################
        'fig_histo_M_ZHMW': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            # elines['ZH_MW_Re_fit'] - elines['alpha_ZH_MW_Re_fit'], r'[Z/H] MW', 3, 4, None,
            # EW_Ha_cen.apply(np.log10), [8, 12, -0.9, 0.3]
            elines['ZH_MW_Re_fit'], r'[Z/H] LW', 3, 8, None,
            EW_Ha_cen.apply(np.log10), [8, 12, -0.7, 0.3]
        ],
        ###############################
        ##########################
        ## MZR colored by EW_Ha ##
        ##########################
        'fig_histo_MZR': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['OH_Re_fit_t2'], r'$12 + \log (O/H)$ t2 ', 2, 8, None,
            EW_Ha_cen.apply(np.log10), [8, 12, 8.3, 9.1]
        ],
        ##########################
        ############################
        ## M-tLW colored by EW_Ha ##
        ############################
        'fig_histo_M_tLW': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['Age_LW_Re_fit'], r'$\log({\rm age/yr})$ LW', 4, 5, None,
            EW_Ha_cen.apply(np.log10), [8, 12, 7.5, 10.5]
        ],
        ############################
        ############################
        ## M-tMW colored by EW_Ha ##
        ############################
        'fig_histo_M_tMW': [
            elines['log_Mass_corr'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 4, 5, None,
            elines['Age_MW_Re_fit'], r'$\log({\rm age/yr})$ MW', 3, 5, None,
            EW_Ha_cen.apply(np.log10), [8, 12, 8.8, 10.2]
        ],
        ############################
        ###############################
        ## Mgas-SFR colored by EW_Ha ##
        ###############################
        'fig_histo_Mgas_SFR': [
            elines['log_Mass_gas_Av_gas_rad'], r'$\log ({\rm M}_{\rm gas,A_V}/{\rm M}_{\odot})$', 6, 5, None,
            elines['lSFR'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', 4, 4, None,
            EW_Ha_cen.apply(np.log10), [5, 11, -4.5, 2.5]
        ],
        ###############################
    }
    for k, v in plot_histo_xy_dict.items():
        print('\n################################')
        print('# %s' % k)
        x, xlabel, n_bins_maj_x, n_bins_min_x, prune_x = v[0:5]
        y, ylabel, n_bins_maj_y, n_bins_min_y, prune_y = v[5:10]
        z = v[10]
        extent = v[11]
        aux_mask = None
        if len(v) == 13:
            aux_mask = v[12]
        output_name = '%s/%s.%s' % (args.figs_dir, k, args.img_suffix)
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        # if k == 'fig_histo_CMD_CUBES' or k == 'fig_histo_CMD_CUBES_NC':
        #     aux_mask = m_redshift
        # else:
        #     aux_mask = None
        plot_histo_xy_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, xlabel=xlabel, xrange=extent[0:2], n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, ylabel=ylabel, yrange=extent[2:4], n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=prune_y, aux_mask=aux_mask)
        if k == 'fig_histo_sSFR_C':
            ax_sc.axvline(x=-11.8, c='k', ls='--')
            ax_sc.axvline(x=-10.8, c='k', ls='--')
        if k == 'fig_histo_M_sSFR':
            ax_sc.axhline(y=-11.8, c='k', ls='--')
            ax_sc.axhline(y=-10.8, c='k', ls='--')
        if k == 'fig_histo_MZR':
            ax_sc = plot_fig_histo_MZR(elines, args, x, y, ax_sc)
        if k == 'fig_histo_M_ZHMW':
            ax_sc = plot_fig_histo_M_ZHMW(elines, args, x, y, ax_sc)
        if k == 'fig_histo_M_tLW' or k == 'fig_histo_M_tMW':
            ax_sc = plot_fig_histo_M_t(elines, args, x, y, ax_sc)
        if k == 'fig_histo_CMD_BR_CUBES_NC':
            plot_text_ax(ax_sc, 'NOCEN', 0.96, 0.95, fs+2, 'top', 'right', 'k')
        if k == 'fig_histo_SFMS_NC':
            plot_text_ax(ax_sc, 'NOCEN', 0.04, 0.95, fs+2, 'top', 'left', 'k')
        # if k == 'fig_histo_M_Mgas':
        #     ax_sc = plot_fig_histo_M_Mgas(elines, args, x, y, ax_sc)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print('################################')
    print('\n################################')
    ################################
    ################################

    ###########
    ## Morph ##
    ###########
    # Create an object spanning only galaxies with defined morphology
    elines_wmorph = elines.loc[elines['morph'] >= 0].copy()
    print('\n#################')
    print('## Morph plots ##')
    print('#################')
    mtI = elines_wmorph['AGN_FLAG'] == 1
    mtII = elines_wmorph['AGN_FLAG'] == 2
    mtBFAGN = (mtI | mtII)
    mtAGN = elines_wmorph['AGN_FLAG'] > 0
    x = elines_wmorph['morph'].apply(morph_adjust)
    H, _ = np.histogram(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtI, _ = np.histogram(x[mtI], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtII, _ = np.histogram(x[mtII], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtAGN, _ = np.histogram(x[mtAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtBFAGN, _ = np.histogram(x[mtBFAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    print('mtp\ttot\ttI\ttII\ttBFAGN\t\ttAGN')
    for mtyp, tot, tI, tII, tBFAGN, tAGN in zip(morph_name[7:], H, HtI, HtII, HtBFAGN, HtAGN):
        print('%s\t%d\t%d\t%d\t%d\t\t%d' % (mtyp, tot, tI, tII, tBFAGN, tAGN))
    ############################
    ## Morph colored by EW_Ha ##
    ############################
    print('\n############################')
    print('## Morph colored by EW_Ha ##')
    print('############################')
    elines_wmorph['u_i']
    plots_dict = {
        'fig_Morph_M': ['log_Mass_corr', [7.5, 12.5], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 6, 2],
        'fig_Morph_C': ['C', [0.5, 5.5], r'${\rm R}90/{\rm R}50$', 6, 2],
        'fig_Morph_SigmaMassCen': ['Sigma_Mass_cen', [1, 5], r'$\log (\Sigma_\star/{\rm M}_{\odot}/{\rm pc}^2)$ cen', 4, 2],
        'fig_Morph_vsigma': ['rat_vel_sigma', [0, 1], r'${\rm V}/\sigma\ ({\rm R} < {\rm Re})$', 2, 5],
        'fig_Morph_Re': ['Re_kpc', [0, 25], r'${\rm Re}/{\rm kpc}$', 6, 2],
        'fig_Morph_sSFR': ['sSFR', [-13.5, -8.5], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2],
        'fig_Morph_bar': ['bar', [-0.2, 2.2], 'bar presence', 3, 1],
        'fig_Morph_BR': ['B_R', [0, 1.5], r'B-R (mag)', 3, 5],
        'fig_Morph_BV': ['B_V', [0, 1.], r'B-V (mag)', 3, 5],
        'fig_Morph_ui': ['u_i', [0, 3.5], r'u-i (mag)', 3, 5],
        'fig_Morph_ur': ['u_r', [0, 3.5], r'u-r (mag)', 3, 5],
    }
    for k, v in plots_dict.items():
        print('\n############################')
        print('# %s' % k)
        ykey, yrange, ylabel, n_bins_maj_y, n_bins_min_y = v
        y = elines_wmorph[ykey]
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        output_name = '%s/%s.%s' % (args.figs_dir, k, args.img_suffix)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        ax_Hx = plot_x_morph(elines=elines_wmorph, args=args, ax=ax_Hx)
        plot_morph_y_colored_by_EW(elines=elines_wmorph, args=args, y=y, ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, ylabel=ylabel, yrange=yrange, n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=None)
        # gs.tight_layout(f)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print('############################')
    print('\n#################')
    ############################

###############################################################################
# END PLOTS ###################################################################
###############################################################################

###############################################################################
# BEGIN IPYTHON RECIPES #######################################################
###############################################################################

###############################
### Concentration histogram ###
###############################
# bins=15
# plt.clf()
# plt.xlabel('R90/R50', fontsize=15)
# plt.ylabel('prob. density', fontsize=15)
# kwargs=dict(histtype='step', linewidth=3, bins=bins, facecolor='none', range=[1.5,4], density=True, align='mid')
# plt.hist(elines.loc[m_et, 'C'], edgecolor='r', label='E+S0+S0a', **kwargs)
# plt.hist(elines.loc[m_lt, 'C'], edgecolor='b', label='spirals+Irr', **kwargs)
# plt.hist(elines.loc[mtI | mtII, 'C'], edgecolor='g', linestyle='--', label='AGN hosts', **kwargs)
# plt.axvline(x=2.6, ls='--', c='k', label='Strateva et al. (2001)')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylim(0, 2)
# plt.gca().xaxis.set_major_locator(MaxNLocator(6))
# plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
# plt.gca().yaxis.set_major_locator(MaxNLocator(4))
# plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
# tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# plt.gca().tick_params(**tick_params)
# plt.legend(fontsize=15)
###############################

###############################################################################
# END IPYTHON RECIPES #########################################################
###############################################################################
