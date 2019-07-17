#!/usr/bin/python3
import os
import sys
import pickle
import itertools
import numpy as np
import seaborn as sns
import matplotlib as mpl
from pytu.lines import Lines
from scipy.stats import describe
from pytu.objects import runstats
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
# latex_column_width = latex_column_width_pt/latex_ppi/1.4
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
scatter_kwargs = dict(s=5, cmap='viridis_r', marker='o', edgecolor='none', alpha=0.6)
# scatter_kwargs = dict(s=5, cmap='viridis_r', vmax=14, vmin=3, marker='o', edgecolor='none', alpha=0.6)
scatter_kwargs_EWmaxmin = dict(s=5, vmax=2.5, vmin=-1, cmap='viridis_r', marker='o', edgecolor='none', alpha=0.6)
#scatter_kwargs_EWmaxmin = dict(c=EW_color.apply(np.log10), s=2, vmax=2.5, vmin=-1, cmap='viridis_r', marker='o', edgecolor='none')
scatter_AGN_tIV_kwargs = dict(s=35, alpha=alpha_AGN_tIV, linewidth=0.5, marker=marker_AGN_tIV, facecolor='none', edgecolor=color_AGN_tIV)
scatter_AGN_tIII_kwargs = dict(s=35, alpha=alpha_AGN_tIII, linewidth=0.5, marker=marker_AGN_tIII, facecolor='none', edgecolor=color_AGN_tIII)
scatter_AGN_tII_kwargs = dict(s=35, linewidth=0.5, marker=marker_AGN_tII, facecolor='none', edgecolor=color_AGN_tII)
scatter_AGN_tI_kwargs = dict(s=35, linewidth=0.5, marker=marker_AGN_tI, facecolor='none', edgecolor=color_AGN_tI)
n_levels_kdeplot = 4

# prop_conf = dict(label=None, extent=None, majloc=None, minloc=None)
props = {
    'log_NII_Ha_cen': dict(fname='logNIIHa_cen', label=r'$\log\ ({\rm [NII]}/{\rm H\alpha})$', extent=[-1.6, 0.8], majloc=3, minloc=5),
    'log_SII_Ha_cen_mean': dict(fname='logSIIHa_cen', label=r'$\log\ ({\rm [SII]}/{\rm H\alpha})$', extent=[-1.6, 0.8], majloc=3, minloc=5),
    'log_OI_Ha_cen_mean': dict(fname='logOIHa_cen', label=r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', extent=[-3, 0.8], majloc=4, minloc=5),
    'log_OIII_Hb_cen_mean': dict(fname='logOIIIHb_cen', label= r'$\log\ ({\rm [OIII]}/{\rm H\beta})$', extent=[-1.2, 1.5], majloc=3, minloc=5),
    'EW_Ha_cen_mean': dict(fname='WHa_cen', label=r'${\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)', extent=[3, 10], majloc=4, minloc=2),
    'EW_Ha_Re': dict(fname='WHa_Re', label=r'${\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)', extent=[3, 10], majloc=4, minloc=2),
    'log_EW_Ha_cen_mean': dict(fname='logWHa_cen', label=r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)', extent=[-1, 2.5], majloc=4, minloc=2),
    'log_EW_Ha_Re': dict(fname='logWHa_Re', label=r'$\log {\rm W}_{{\rm H}\alpha}^{\rm Re}$ (\AA)', extent=[-1, 2.5], majloc=4, minloc=2),
    'C': dict(fname='C', label=r'${\rm R}90/{\rm R}50$', extent=[0.5, 4.5], majloc=4, minloc=2),
    'log_Mass_corr': dict(fname='M', label=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', extent=[8, 12], majloc=4, minloc=5),
    'log_Mass_corr_NC': dict(fname='M_NC', label=r'$\log ({\rm M}_\star^{\rm NC}/{\rm M}_{\odot})$', extent=[8, 12], majloc=4, minloc=5),
    'log_Mass_gas_Av_gas_rad': dict(fname='Mgas', label=r'$\log ({\rm M}_{\rm gas,A_V}/{\rm M}_{\odot})$', extent=[4.8, 10.2], majloc=6, minloc=5),
    'lSFR': dict(fname='SFRHa', label=r'$\log ({\rm SFR}_{\rm H\alpha}/{\rm M}_{\odot}/{\rm yr})$', extent=[-4.5, 2.5], majloc=4, minloc=5),
    'lSFR_NC': dict(fname='SFRHa_NC', label=r'$\log ({\rm SFR}_{\rm H\alpha}^{\rm NC}/{\rm M}_{\odot}/{\rm yr})$', extent=[-4.5, 2.5], majloc=4, minloc=5),
    'log_SFR_SF': dict(fname='SFRHaSF', label=r'$\log ({\rm SFR}_{\rm H\alpha}^{\rm SF}/{\rm M}_{\odot}/{\rm yr})$', extent=[-4.5, 2.5], majloc=4, minloc=5),
    'log_SFR_ssp': dict(fname='SFRssp', label=r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', extent=[-4.5, 2.5], majloc=4, minloc=5),
    'sSFR': dict(fname='sSFRHa', label=r'$\log ({\rm sSFR}_{\rm H\alpha}/{\rm yr})$', extent=[-13.5, -8.5], majloc=5, minloc=2),
    'sSFR_SF': dict(fname='sSFRHaSF', label=r'$\log ({\rm sSFR}_{\rm H\alpha}^{\rm SF}/{\rm yr})$', extent=[-14.5, -8.5], majloc=6, minloc=2),
    'sSFR_ssp': dict(fname='sSFRssp', label=r'$\log ({\rm sSFR}_\star/{\rm yr})$', extent=[-12.5, -8.5], majloc=4, minloc=2),
    'SFE': dict(fname='SFEHa', label=r'$\log$ (${\rm SFE}_{\rm H\alpha}$/yr)', extent=[-10, -5], majloc=3, minloc=4),
    'SFE_SF': dict(fname='SFEHaSF', label=r'$\log$ (${\rm SFE}_{\rm SF}$/yr)', extent=[-11, -6], majloc=3, minloc=4),
    'SFE_ssp': dict(fname='SFEssp', label=r'$\log$ (${\rm SFE}_\star$/yr)', extent=[-10, -5], majloc=3, minloc=4),
    'log_fgas': dict(fname='logfgas', label=r'$\log\ f_{\rm gas}$', extent=[-5, 0], majloc=5, minloc=2),
    'g_r': dict(fname='gr', label=r'g-r (mag)', extent=[0, 1], majloc=5, minloc=2),
    'g_r_NC': dict(fname='gr_NC', label=r'${\rm g-r}^{\rm NC}$ (mag)', extent=[0, 1], majloc=5, minloc=2),
    'u_r': dict(fname='ur', label=r'u-r (mag)', extent=[0, 3.5], majloc=3, minloc=2),
    'u_r_NC': dict(fname='ur_NC', label=r'${\rm u-r}^{\rm NC}$ (mag)', extent=[0, 3.5], majloc=3, minloc=2),
    'u_i': dict(fname='ui', label=r'u-i (mag)', extent=[0, 3.5], majloc=3, minloc=2),
    'u_i_NC': dict(fname='ui_NC', label=r'${\rm u-i}^{\rm NC}$ (mag)', extent=[0, 3.5], majloc=3, minloc=2),
    'B_R': dict(fname='BR', label=r'B-R (mag)', extent=[0, 1.5], majloc=3, minloc=5),
    'B_R_NC': dict(fname='BR_NC', label=r'${\rm B-R}^{\rm NC}$ (mag)', extent=[0, 1.5], majloc=3, minloc=5),
    'B_V': dict(fname='BV', label=r'B-V (mag)', extent=[0, 1], majloc=5, minloc=2),
    'B_V_NC': dict(fname='BV_NC', label=r'${\rm B-V}^{\rm NC}$ (mag)', extent=[0, 1], majloc=5, minloc=2),
    'Mabs_r': dict(fname='Mr', label=r'${\rm M}_{\rm r}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_r_NC': dict(fname='Mr_NC', label=r'${\rm M}_{\rm r}^{\rm NC}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_i': dict(fname='Mi', label=r'${\rm M}_{\rm i}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_i_NC': dict(fname='Mi_NC', label=r'${\rm M}_{\rm i}^{\rm NC}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_R': dict(fname='MR', label=r'${\rm M}_{\rm R}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_R_NC': dict(fname='MR_NC', label=r'${\rm M}_{\rm R}^{\rm NC}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_V': dict(fname='MV', label=r'${\rm M}_{\rm V}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'Mabs_V_NC': dict(fname='MV_NC', label=r'${\rm M}_{\rm V}^{\rm NC}$ (mag)', extent=[-24, -15], majloc=5, minloc=2),
    'ZH_MW_Re_fit': dict(fname='ZHMW', label=r'[Z/H] MW', extent=[-0.7, 0.3], majloc=3, minloc=3),
    'ZH_LW_Re_fit': dict(fname='ZHLW', label=r'[Z/H] LW', extent=[-0.7, 0.3], majloc=3, minloc=3),
    'OH_Re_fit_t2': dict(fname='OHt2', label=r'$12 + \log (O/H)$ t2', extent=[8.3, 9.1], majloc=4, minloc=2),
    'Age_LW_Re_fit': dict(fname='tLW', label=r'$\log({\rm age/yr})$ LW', extent=[7.5, 10.5], majloc=3, minloc=5),
    'Age_MW_Re_fit': dict(fname='tMW', label=r'$\log({\rm age/yr})$ MW', extent=[8.8, 10.2], majloc=3, minloc=5),
    'log_age_mean_LW': dict(fname='tmLW', label=r'$\log({\rm age/yr})$ LW', extent=[7.5, 10.5], majloc=3, minloc=5),
    'NUV_r_SDSS': dict(fname='NUV_r_SDSS', label=r'NUV-r (mag)', extent=[0, 7], majloc=7, minloc=2),
    'NUV_r_CUBES': dict(fname='NUV_r_CUBES', label=r'NUV-r (mag)', extent=[0, 7], majloc=7, minloc=2),
    'Sigma_Mass_cen': dict(fname='mu_cen', label=r'$\log (\Sigma_\star^{\rm cen}/{\rm M}_{\odot}/{\rm pc}^2)$', extent=[1, 5], majloc=4, minloc=5),
    'bar': dict(fname='bar', label='bar presence', extent=[-0.2, 2.2], majloc=3, minloc=1),
    'rat_vel_sigma': dict(fname='vsigma', label=r'${\rm V}/\sigma\ ({\rm R} < {\rm Re})$', extent=[0, 1], majloc=5, minloc=2),
    'Re_kpc': dict(fname='Re', label=r'${\rm Re}/{\rm kpc}$', extent=[0, 20], majloc=5, minloc=4),
}


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
        'debug': False,
    }

    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--debug', '-d', action = 'store_true', default = default_args['debug'])
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


def modlogOHSF2017_O3N2(x):
    a = 8.53
    b = 0.003
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


def xyz_clean_sort_interval(x, y, z=None, interval=None, mask=None):
    if z is None:
        sel = np.isfinite(x) & np.isfinite(y)
        # sel = ~(np.isnan(x) | np.isnan(y))
    else:
        sel = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        # sel = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if interval is not None:
        sel &= (x > interval[0]) & (x < interval[1]) & (y > interval[2]) & (y < interval[3])
    X = x[sel]
    Y = y[sel]
    iS = np.argsort(X)
    XS = X[iS]
    YS = Y[iS]
    if z is not None:
        Z = z[sel]
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
    from scipy.stats import pearsonr, spearmanr
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
        xm, ym = ma_mask_xyz(XS[sel_c], YS[sel_c])
        c_p_c = pearsonr(xm.compressed(), ym.compressed())[0]
        c_r_c = spearmanr(xm.compressed(), ym.compressed())[0]
        print('pearsonr:%.2f spearmanr:%.2f' % (pearsonr(xm.compressed(), ym.compressed())[0], spearmanr(xm.compressed(), ym.compressed())[0]))
    xm, ym = ma_mask_xyz(x_bins_center__r, YS__r)
    p = np.ma.polyfit(xm.compressed(), ym.compressed(), 1)
    print('linear regression with no sigma clip:')
    print(p)
    sigma_dev = (y - np.polyval(p, x)).std()
    print('sigma dev = {:.5f}'.format(sigma_dev))
    xm, ym = ma_mask_xyz(XS, YS)
    c_p = pearsonr(xm.compressed(), ym.compressed())[0]
    c_r = spearmanr(xm.compressed(), ym.compressed())[0]
    print('pearsonr:%.2f spearmanr:%.2f' % (c_p, c_r))
    return p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c


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


def plot_WHAN(args, N2Ha, WHa, z=None, f=None, ax=None, extent=None, output_name=None, cmap='viridis_r', mask=None, N=False, z_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True):
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


def plot_colored_by_z(elines, args, x, y, z, xlabel=None, ylabel=None, zlabel=None, extent=None, n_bins_maj_x=5, n_bins_maj_y=5, n_bins_min_x=5, n_bins_min_y=5, prune_x='upper', prune_y=None, output_name=None, markAGNs=False, f=None, ax=None, sc_kwargs=None, z_maxlocator=4, z_extent=None):
    if z_extent is None:
        z_extent = [-1, 2.5]
    if zlabel is None:
        zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
    if sc_kwargs is None:
        sc_kwargs = scatter_kwargs
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    mXYZ = (x.notna() & y.notna() & z.notna())
    print('x:%s:%d  y:%s:%d  z:%s:%d  all:%d  tIAGN:%d  tIIAGN:%d  AGN:%d' % (x.name, x.notna().sum(), y.name, y.notna().sum(), z.name, z.notna().sum(), mXYZ.sum(), (mXYZ & mtI).sum(), (mXYZ & mtII).sum(), (mXYZ & mtAGN).sum()))
    # bottom, top, left, right = 0.30, 0.95, 0.2, 0.75
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    if f is None:
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
    if args.debug:
        txt = '%d:%d:%d' % (mXYZ.astype('int').sum(), (mXYZ & mtI).astype('int').sum(), (mXYZ & mtII).astype('int').sum())
        plot_text_ax(ax, txt, 0.96, 0.95, args.fontsize+2, 'top', 'right', 'k')
    sc = ax.scatter(x, y, c=z, vmin=z_extent[0], vmax=z_extent[1], **sc_kwargs)
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
    cb.solids.set(alpha=1)
    cb.set_label(zlabel, fontsize=args.fontsize+1)
    cb.locator = MaxNLocator(z_maxlocator)
    # cb_ax.minorticks_on()
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
    if extent is not None:
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(n_bins_maj_x, prune=prune_x))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n_bins_min_x))
    ax.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
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


def plot_histo_xy_colored_by_z(elines, args, x, y, z, ax_Hx, ax_Hy, ax_sc, xlabel=None, xrange=None, n_bins_maj_x=5, n_bins_min_x=5, prune_x=None, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, aux_mask=None, zlabel=None, z_extent=None):
    if z_extent is None:
        z_extent = [-1, 2.5]
    if zlabel is None:
        zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
    if aux_mask is not None:
        elines = elines.loc[aux_mask]
    mtI = (elines['AGN_FLAG'] == 1)
    mtII = (elines['AGN_FLAG'] == 2)
    mtAGN = mtI | mtII
    # mtIII = elines['AGN_FLAG'] == 3
    mXYZ = (x.notna() & y.notna() & z.notna())
    if args.debug:
        txt = '%d:%d:%d' % (mXYZ.astype('int').sum(), (mXYZ & mtI).astype('int').sum(), (mXYZ & mtII).astype('int').sum())
        plot_text_ax(ax_sc, txt, 0.96, 0.95, args.fontsize+2, 'top', 'right', 'k')
    print('x:%s:%d  y:%s:%d  z:%s:%d  all:%d  tIAGN:%d  tIIAGN:%d  AGN:%d' % (x.name, x.notna().sum(), y.name, y.notna().sum(), z.name, z.notna().sum(), mXYZ.sum(), (mXYZ & mtI).sum(), (mXYZ & mtII).sum(), (mXYZ & mtAGN).sum()))
    Nbins = 20
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
    sc = ax_sc.scatter(x, y, c=z, vmin=z_extent[0], vmax=z_extent[1], **scatter_kwargs)
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
    cb.solids.set(alpha=1)
    cb.set_label(zlabel, fontsize=args.fontsize+2)
    cb.locator = MaxNLocator(4)
    # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
    cb_ax.tick_params(which='both', direction='in')
    cb.update_ticks()
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


def plot_morph_y_colored_by_z(elines, args, y, z, ax_Hx, ax_Hy, ax_sc, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, zlabel=None, z_extent=None):
    if z_extent is None:
        z_extent = [-1, 2.5]
    if zlabel is None:
        zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
    # EW_color = elines['EW_Ha_cen_mean'].apply(np.abs)
    # scatter_kwargs_EWmaxmin = dict(c=EW_color.apply(np.log10), s=2, vmax=2.5, vmin=-1, cmap='viridis_r', marker='o', edgecolor='none')
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    morph = elines['morph'].apply(morph_adjust)
    mXYZ = (morph.notna() & y.notna() & z.notna())
    if args.debug:
        txt = '%d:%d:%d' % (mXYZ.astype('int').sum(), (mXYZ & mtI).astype('int').sum(), (mXYZ & mtII).astype('int').sum())
        plot_text_ax(ax, txt, 0.96, 0.95, args.fontsize+2, 'top', 'right', 'k')
    print('x:morph:%d  y:%s:%d  z:%s:%d  all:%d  tIAGN:%d  tIIAGN:%d  AGN:%d' % (morph.notna().sum(), y.name, y.notna().sum(), z.name, z.notna().sum(), mXYZ.sum(), (mXYZ & mtI).sum(), (mXYZ & mtII).sum(), (mXYZ & mtAGN).sum()))
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
    data = [y.loc[(morph == mt) & y.notna()] for mt in m]
    d = ax_sc.boxplot(data,
                      showfliers=False, notch=False, positions=m,
                      flierprops=dict(linewidth=0.2, marker='D', markersize=1, color='r'),
                      boxprops=dict(linewidth=0.2),
                      whiskerprops=dict(linewidth=0.2),
                      capprops=dict(linewidth=0.2),
                      medianprops=dict(linewidth=0, color='k'))
    cmap = mpl.cm.get_cmap('viridis_r')
    norm = mpl.colors.Normalize(vmin=-1, vmax=2.5)
    colors_list = [cmap(norm(elines.loc[morph == mt, 'log_EW_Ha_Re'].mean())) for mt in m]
    # color boxes in boxplot
    i = 0
    for i in range(len(colors_list)):
        box = d['boxes'][i]
        box_coords = np.column_stack([box.get_xdata(), box.get_ydata()])
        ax_sc.add_patch(mpl.patches.Polygon(box_coords, facecolor=colors_list[i], zorder=0.01))
    tmp_kw = scatter_kwargs.copy()
    tmp_kw['alpha'] = 0
    sc = ax_sc.scatter(morph, y, c=z, vmin=z_extent[0], vmax=z_extent[1], **tmp_kw)
    # mALLAGN = (elines['AGN_FLAG'] > 0)
    # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
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
    # cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
    cb.solids.set(alpha=1)
    cb.set_label(zlabel, fontsize=args.fontsize+2)
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


def plot_fig_histo_MZR_t2(elines, args, x, y, ax):
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
    # from scipy.optimize import curve_fit
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
    print(elines.loc[m_y_tI_above.index[m_y_tI_above], [x.name, y.name, 'modlogOHSF2017_t2']])
    print('AGN Type II:')
    print(elines.loc[m_y_tII_above.index[m_y_tII_above], [x.name, y.name, 'modlogOHSF2017_t2']])
    # print(elines.loc[m_y_tIII_above.index[m_y_tIII_above], [x.name, y.name, 'modlogOHSF2017_t2']])
    print('AGNs:')
    print(elines.loc[m_y_tAGN_above.index[m_y_tAGN_above], [x.name, y.name, 'modlogOHSF2017_t2']])
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


def plot_fig_histo_MZR_O3N2(elines, args, x, y, ax):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    ### MZR ###
    mXnotnan = ~np.isnan(x)
    X = x.loc[mXnotnan].values
    iS = np.argsort(X)
    XS = X[iS]
    elines['modlogOHSF2017_O3N2'] = modlogOHSF2017_O3N2(elines['log_Mass_corr'])
    YS = (elines['modlogOHSF2017_O3N2'].loc[mXnotnan].values)[iS]
    mY = YS >= 8
    mX = XS <= 11.5
    ax.plot(XS[mX & mY], YS[mX & mY], 'k-')
    ### best-fit ###
    from scipy.optimize import curve_fit
    mnotnan = ~(np.isnan(x) | np.isnan(y)) & (x <= 11.5) & (x >= 8)
    XFIT = x.loc[mnotnan].values
    iSFIT = np.argsort(XFIT)
    XFITS = XFIT[iSFIT]
    YFIT = y.loc[mnotnan].values
    YFITS = YFIT[iSFIT]
    popt, pcov = curve_fit(f=modlogOHSF2017, xdata=XFITS, ydata=YFITS, p0=[8.5, 0.005, 11.5], bounds=[[8.4, 0.001, 11.499], [9, 0.022, 11.501]])
    ax.plot(XFITS, modlogOHSF2017(XFITS, *popt), 'k--')
    print('a:%.2f b:%.4f c:%.1f' % (popt[0], popt[1], popt[2]))
    ### Above ###
    m_y_tI_above = y.loc[mtI] > x.loc[mtI].apply(modlogOHSF2017_O3N2)
    m_y_tII_above = y.loc[mtII] > x.loc[mtII].apply(modlogOHSF2017_O3N2)
    # m_y_tIII_above = y.loc[mtIII] > x.loc[mtIII].apply(modlogOHSF2017_O3N2)
    m_y_tAGN_above = y.loc[mtAGN] > x.loc[mtAGN].apply(modlogOHSF2017_O3N2)
    print('AGN Type I:')
    print(elines.loc[m_y_tI_above.index[m_y_tI_above], [x.name, y.name, 'modlogOHSF2017_O3N2']])
    print('AGN Type II:')
    print(elines.loc[m_y_tII_above.index[m_y_tII_above], [x.name, y.name, 'modlogOHSF2017_O3N2']])
    # print(elines.loc[m_y_tIII_above.index[m_y_tIII_above], [x.name, y.name, 'modlogOHSF2017_O3N2']])
    print('AGNs:')
    print(elines.loc[m_y_tAGN_above.index[m_y_tAGN_above], [x.name, y.name, 'modlogOHSF2017_O3N2']])
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
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    mtIII = elines['AGN_FLAG'] == 3
    mtIV = elines['AGN_FLAG'] == 4
    mtAGN = mtI | mtII
    mtallAGN = mtI | mtII | mtIII | mtIV

    y_AGNs_mean = y.loc[mtallAGN].mean()
    y_AGNs_mean_st = (10**y.loc[mtallAGN]).mean()
    y_BF_AGNs_mean = y.loc[mtAGN].mean()
    y_BF_AGNs_mean_st = (10**y.loc[mtAGN]).mean()
    y_AGNs_tI_mean = y.loc[mtI].mean()
    y_AGNs_tI_mean_st = (10**y.loc[mtI]).mean()
    y_AGNs_tII_mean = y.loc[mtII].mean()
    y_AGNs_tII_mean_st = (10**y.loc[mtII]).mean()
    print('logstat: y_AGNs_mean: %.2f Gyr - %s' % (10**(y_AGNs_mean - 9), describe(y.loc[mtallAGN])))
    print('y_AGNs_mean: %.2f Gyr - %s' % (y_AGNs_mean_st/1e9, describe(10**y.loc[mtallAGN])))
    print('logstat: y_BF_AGNs_mean: %.2f Gyr - %s' % (10**(y_BF_AGNs_mean - 9), describe(y.loc[mtAGN])))
    print('y_BF_AGNs_mean: %.2f Gyr - %s' % (y_BF_AGNs_mean_st/1e9, describe(10**y.loc[mtAGN])))
    print('logstat: y_AGNs_tI_mean: %.2f Gyr - %s' % (10**(y_AGNs_tI_mean - 9), describe(y.loc[mtI])))
    print('y_AGNs_tI_mean: %.2f Gyr - %s' % (y_AGNs_tI_mean_st/1e9, describe(10**y.loc[mtI])))
    print('logstat: y_AGNs_tII_mean: %.2f Gyr - %s' % (10**(y_AGNs_tII_mean - 9), describe(y.loc[mtII])))
    print('y_AGNs_tII_mean: %.2f Gyr - %s' % (y_AGNs_tII_mean_st/1e9, describe(10**y.loc[mtII])))
    ax_sc.axhline(y_BF_AGNs_mean, c='g', ls='--')
    ax_sc.text(0.05, 0.85, '%.2f Gyr' % (10**(y_BF_AGNs_mean - 9)), color='g', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    # print('y_AGNs_tIII_mean: %.2f Gyr' % 10**(y_AGNs_tIII_mean - 9))
    WHa = elines['EW_Ha_cen_mean']
    WHa_Re = elines['EW_Ha_Re']
    WHa_ALL = elines['EW_Ha_ALL']
    m_dict_WHa = dict(cen=WHa, Re=WHa_Re, ALL=WHa_ALL)
    m_dict = {k: dict() for k in m_dict_WHa.keys()}
    logt_dict = {k: dict() for k in m_dict_WHa.keys()}
    for k1, v1 in m_dict_WHa.items():
        m = ~(np.isnan(x) | np.isnan(y) | np.isnan(v1))
        m_dict[k1] = dict(hDIG=v1 <= args.EW_hDIG,
                          OLD=v1 < 6, YOUNG=v1 > 6,
                          GV=(v1 > args.EW_hDIG) & (v1 <= args.EW_SF),
                          SFc=v1 > args.EW_SF,)
        for k2, v2 in m_dict[k1].items():
            logt = y.loc[m & v2]
            t = 10**y.loc[m & v2]
            logt_mean = logt.mean()
            t_mean = t.mean()
            logt_std = logt.std()
            print('logstat: %s: %s: %.2f Gyr - %s' % (k1, k2, 10**(logt_mean - 9), describe(logt)))
            print('stat: %s: %s: %.2f Gyr - %s' % (k1, k2, t_mean/1e9, describe(t)))
            logt_dict[k1][k2] = logt
    t = logt_dict['Re']['SFc'].mean()
    ax_sc.axhline(t, c='b', ls='--')
    ax_sc.text(0.05, 0.77, '%.2f Gyr' % (10**(t - 9)), color='b', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    t = logt_dict['Re']['hDIG'].mean()
    ax_sc.axhline(t, c='r', ls='--')
    ax_sc.text(0.05, 0.93, '%.2f Gyr' % (10**(t - 9)), color='r', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    ## Using the SFMS for YOUNG mean age calculation
    SFRHa = elines['log_SFR_SF']
    for k_WHa, v in m_dict_WHa.items():
        for k_SF in ['YOUNG', 'SFc']:
            m = m_dict[k_WHa][k_SF]
            logMass = x.loc[m]
            logt = y.loc[m]
            logSFRHa = SFRHa.loc[m]
            XS, YS, ZS = xyz_clean_sort_interval(logMass.values, logt.values, logSFRHa.values)
            if interval is None:
                interval = [8.3, 11.8, 7.5, 10.5]
            x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
            ZS_c__r, N_c__r, sel_c, ZS__r, N__r, sel = redf_xy_bins_interval(XS, ZS, x_bins__r, clip=2, interval=interval)
            p = np.ma.polyfit(x_bins_center__r, ZS_c__r, 1)
            print(p)
            print('\t%s: %s: %.2f Gyr - %.2f Gyr' % (k_WHa, k_SF, 10**(YS[sel].mean()-9), 10**(YS[sel_c].mean()-9)))
    # ### SFG ###
    # SFRHa = elines['lSFR']
    # x_SF = x.loc[YOUNG]
    # y_SF = y.loc[YOUNG]
    # SFRHa_SF = SFRHa.loc[YOUNG]
    # XS_SF, YS_SF, SFRHaS_SF = xyz_clean_sort_interval(x_SF.values, y_SF.values, SFRHa_SF.values)
    # if interval is None:
    #     interval = [8.3, 11.8, 7.5, 10.5]
    # x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # # SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, interval=interval)
    # SFRHaS_SF_c__r, N_c__r, sel_c, SFRHaS_SF__r, N__r, sel = redf_xy_bins_interval(XS_SF, SFRHaS_SF, x_bins__r, clip=2, interval=interval)
    # p = np.ma.polyfit(x_bins_center__r, SFRHaS_SF_c__r, 1)
    # print(p)
    # print(10**(YS_SF[sel].mean()-9), 10**(YS_SF[sel_c].mean()-9))
    # mean_t_SF = YS_SF[sel].mean()
    # # ax_sc.axhline(mean_t_SF, xmin=0.9/(12.-8.), c='b', ls='--')
    # ax_sc.axhline(mean_t_SF, c='b', ls='--')
    # ax_sc.text(0.05, 0.77, '%.2f Gyr' % (10**(mean_t_SF - 9)), color='b', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    # ### RG ###
    # # x_hDIG = x.loc[OLD]
    # # y_hDIG = y.loc[OLD]
    # # SFRHa_hDIG = SFRHa.loc[OLD]
    # # XS_hDIG, YS_hDIG, SFRHaS_hDIG = xyz_clean_sort_interval(x_hDIG.values, y_hDIG.values, SFRHa_hDIG.values)
    # # if interval is None:
    # #     interval = [8.3, 11.8, 7.5, 10.5]
    # # x_bins__r, x_bins_center__r, nbins = create_bins(interval[0:2], 0.1)
    # # # SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, interval=interval)
    # # SFRHaS_hDIG_c__r, N_c__r, sel_c, SFRHaS_hDIG__r, N__r, sel = redf_xy_bins_interval(XS_hDIG, SFRHaS_hDIG, x_bins__r, clip=2, interval=interval)
    # # p = np.ma.polyfit(x_bins_center__r, SFRHaS_hDIG_c__r, 1)
    # # print(p)
    # # print(10**(YS_hDIG[sel].mean()-9), 10**(YS_hDIG[sel_c].mean()-9))
    # # mean_t_hDIG = YS_hDIG[sel].mean()
    # mean_t_hDIG = y_OLD_mean
    # # # ax_sc.axhline(mean_t_hDIG, xmin=0.9/(12.-8.), c='r', ls='--')
    # ax_sc.axhline(mean_t_hDIG, c='r', ls='--')
    # ax_sc.text(0.05, 0.93, '%.2f Gyr' % (10**(mean_t_hDIG - 9)), color='r', fontsize=args.fontsize, va='center', transform=ax.transAxes)
    return ax_sc

def plot_fig_histo_M_fgas(elines, args, x, y, ax, interval=None):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    WHa = elines['EW_Ha_cen_mean']
    WHa_Re = elines['EW_Ha_Re']
    WHa_ALL = elines['EW_Ha_ALL']
    if interval is None:
        interval = [9, 11.5, -5, 0]
    XS, YS, ZS = xyz_clean_sort_interval(x.values, y.values, WHa_Re.values)
    x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.5)
    _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, interval)
    print(y_mean, N_y_mean)
    ax.plot(x_bincenter__r, y_mean, 'k--')
    SFc = ZS > args.EW_SF
    GV = (ZS > args.EW_hDIG) & (ZS <= args.EW_SF)
    RG = ZS <= args.EW_hDIG
    _, _, _, y_mean_SF, N_y_mean_SF, _ = redf_xy_bins_interval(XS[SFc], YS[SFc], x_bins__r, interval)
    print(y_mean_SF, N_y_mean_SF)
    p = np.ma.polyfit(x_bincenter__r, y_mean_SF, 1)
    ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='b', label='SFG')
    # _, _, _, y_mean_RG, N_y_mean_RG, _ = redf_xy_bins_interval(XS[RG], YS[RG], x_bins__r, interval)
    # print(y_mean_RG, N_y_mean_RG)
    # p = np.ma.polyfit(x_bincenter__r, y_mean_RG, 1)
    # print(y_mean_RG, p)
    # ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='r', label='RG')
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


def plot_fig_histo_M_SFE(elines, args, x, y, ax, interval=None):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    WHa = elines['EW_Ha_cen_mean']
    WHa_Re = elines['EW_Ha_Re']
    WHa_ALL = elines['EW_Ha_ALL']
    if interval is None:
        interval = [9, 11.5, -11, -6]
    XS, YS, ZS = xyz_clean_sort_interval(x.values, y.values, WHa_Re.values)
    x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.5)
    _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, interval)
    print(y_mean, N_y_mean)
    ax.plot(x_bincenter__r, y_mean, 'k--')
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


def plot_RSB(elines, args, x, y, ax, interval=None):
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    # mtIII = elines['AGN_FLAG'] == 3
    mtAGN = mtI | mtII
    WHacen = elines['EW_Ha_cen_mean']
    WHaRe = elines['EW_Ha_Re']
    SFG = WHaRe > args.EW_SF
    GVG = (WHaRe > args.EW_hDIG) & (WHaRe <= args.EW_SF)
    RG = WHaRe <= args.EW_hDIG
    masks = dict(SFG=SFG, GVG=GVG, RG=RG)
    colors = dict(SFG='b', GVG='g', RG='r')
    for k, v in masks.items():
        c = colors[k]
        print(k,c)
        m = v & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        xm, ym, zm = ma_mask_xyz(x.loc[m].values, y.loc[m].values, z.loc[m].values)
        rs = runstats(xm.compressed(), ym.compressed(),
                      smooth=True, sigma=1.2,
                      debug=True, gs_prc=True,
                      poly1d=True)
        print(rs.poly1d_median_slope, rs.poly1d_median_intercept)
        # p = [rs.poly1d_median_slope, rs.poly1d_median_intercept]
        # ax.plot(xm.compressed(), np.polyval(p, xm.compressed()), color=c, lw=1)
        ax.plot(rs.xS, rs.yS, color=c, lw=1)
        # XS, YS, ZS = xyz_clean_sort_interval(x.loc[v].values, y.loc[v].values, WHa_Re.loc[v].values)
        # nbins = 25
        # step = (interval[1] - interval[0]) / nbins
        # # step = 0.3
        # x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], step)
        # _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, interval)
        # # print(y_mean, N_y_mean)
        #ax.plot(x_bincenter__r, y_mean, '%c--' % colors[k])
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
    args.bug = bug
    args.EW_SF = EW_SF
    args.EW_AGN = EW_AGN
    args.EW_hDIG = EW_hDIG
    args.EW_strong = EW_strong
    args.EW_verystrong = EW_verystrong
    args.props = props
    debug_var(True, args=args)

    sel_NIIHa = pickled['sel_NIIHa']
    sel_OIIIHb = pickled['sel_OIIIHb']
    sel_SIIHa = pickled['sel_SIIHa']
    sel_OIHa = pickled['sel_OIHa']
    sel_MS = pickled['sel_MS']
    sel_EW_cen = pickled['sel_EW_cen']
    sel_EW_ALL = pickled['sel_EW_ALL']
    sel_EW_Re = pickled['sel_EW_Re']
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
    EW_Ha_Re = elines['EW_Ha_Re']
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
        Line2D([0], [0], marker=marker_AGN_tI, markeredgecolor=color_AGN_tI, label='Type-I (%d)' % N_AGN_tI, markerfacecolor='none', markersize=7, markeredgewidth=0.4, linewidth=0),
        Line2D([0], [0], marker=marker_AGN_tII, markeredgecolor=color_AGN_tII, label='Type-II (%d)' % N_AGN_tII, markerfacecolor='none', markersize=7, markeredgewidth=0.4, linewidth=0),
        # Line2D([0], [0], marker=marker_AGN_tIII, alpha=alpha_AGN_tIII, markeredgecolor=color_AGN_tIII, label=r'by [NII]/H$\alpha$ and other (+%d)' % N_AGN_tIII, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
        # Line2D([0], [0], marker=marker_AGN_tIV, alpha=alpha_AGN_tIV, markeredgecolor=color_AGN_tIV, label=r'by [NII]/H$\alpha$ (+%d)' % N_AGN_tIV, markerfacecolor='none', markersize=5, markeredgewidth=0.1, linewidth=0),
    ]

    sSFR_SF = elines['log_SFR_SF'] - elines['log_Mass_corr']
    sSFR = elines['lSFR'] - elines['log_Mass_corr']
    sSFR_ssp = elines['log_SFR_ssp'] - elines['log_Mass_corr']
    Mrat = 10**(elines['log_Mass_corr'] - elines['log_Mass_gas_Av_gas_rad'])
    fgas = 1 / (1 + Mrat)

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
    sc = ax.scatter(x, y, c=elines['log_EW_Ha_cen_mean'], **scatter_kwargs_EWmaxmin)
    # ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    # ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
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
    ax.legend(handles=legend_elements, ncol=1, loc=2, frameon=False, fontsize=6, borderpad=0, borderaxespad=0.75)
    ##########################
    # SII/Ha
    ##########################
    print('##########################')
    print('## [SII]/Ha             ##')
    print('##########################')
    ax = ax1
    x = log_SII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=elines['log_EW_Ha_cen_mean'], **scatter_kwargs_EWmaxmin)
    # ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    # ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
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
    sc = ax.scatter(x, y, c=elines['log_EW_Ha_cen_mean'], **scatter_kwargs_EWmaxmin)
    # ax.scatter(x.loc[mtIV], y.loc[mtIV], **scatter_AGN_tIV_kwargs)
    # ax.scatter(x.loc[mtIII], y.loc[mtIII], **scatter_AGN_tIII_kwargs)
    ax.scatter(x.loc[mtII], y.loc[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y.loc[mtI], **scatter_AGN_tI_kwargs)
    ax.set_xlabel(r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', fontsize=fs+4)
    cb_ax = f.add_axes([right, bottom, 0.02, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.solids.set(alpha=1)
    cb.set_label(r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)', fontsize=fs+4)
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
    y = elines['EW_Ha_cen_mean']
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
    y = elines['log_EW_Ha_cen_mean']
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
    output_name = '%s/fig_WHAN.%s' % (args.figs_dir, args.img_suffix)
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('##################################')
    ##################################

    ###############################
    # ALL plots use the same z-axis
    ###############################
    z_key = 'log_EW_Ha_Re'
    # z_key = 'EW_Ha_Re'
    # z_key = 'log_EW_Ha_cen_mean'
    z = elines[z_key]
    z_label = props[z_key]['label']
    z_extent = props[z_key]['extent']
    ###############################
    ###############################
    y_key_list = ['lSFR', 'lSFR_NC', 'log_SFR_SF', 'log_SFR_ssp', 'log_fgas', 'SFE', 'SFE_SF', 'SFE_ssp']
    for y_key in y_key_list:
        print('\n###################################')
        x_key = 'log_Mass_corr'
        if y_key[-3::] == '_NC':
            x_key = 'log_Mass_corr_NC'
        x = elines[x_key]
        x_label = props[x_key]['label']
        x_extent = props[x_key]['extent']
        x_majloc = props[x_key]['majloc']
        x_minloc = props[x_key]['minloc']
        k = '%s_%s' % (props[x_key]['fname'], props[y_key]['fname'])
        fname = 'fig_%s' % k
        print('# %s' % fname)
        print('###################################')
        y = elines[y_key]
        y_label = props[y_key]['label']
        y_extent = props[y_key]['extent']
        y_majloc = props[y_key]['majloc']
        y_minloc = props[y_key]['minloc']
        extent = x_extent + y_extent
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
        plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                          xlabel=x_label, ylabel=y_label, extent=extent,
                          n_bins_maj_x=x_majloc, n_bins_min_x=x_minloc,
                          n_bins_maj_y=y_majloc, n_bins_min_y=y_minloc,
                          prune_x=None, zlabel=z_label, z_extent=z_extent,
                          f=f, ax=ax)
        if y_key[-3::] == '_NC':
            plot_text_ax(ax, 'NOCEN', 0.04, 0.95, args.fontsize+2, 'top', 'left', 'k')
        # xm, ym = ma_mask_xyz(x, y, mask=~mALLAGN)
        # sns.kdeplot(xm.compressed(), ym.compressed(), ax=ax, color='red', n_levels=n_levels_kdeplot, alpha=0.4)
        WHa = EW_Ha_cen
        WHa_Re = elines['EW_Ha_Re']
        WHa_ALL = elines['EW_Ha_ALL']
        hDIG = sel_EW_cen & (WHa <= args.EW_hDIG)
        SFc = sel_EW_cen & (WHa > args.EW_SF)
        hDIG_Re = sel_EW_Re & (WHa_Re <= args.EW_hDIG)
        SFc_Re = sel_EW_Re & (WHa_Re > args.EW_SF)
        hDIG_ALL = sel_EW_ALL & (WHa_ALL <= args.EW_hDIG)
        SFc_ALL = sel_EW_ALL & (WHa_ALL > args.EW_SF)
        interval = [8.3, 11.8, 7.5, 10.5]
        dict_masks = dict(hDIG=hDIG, hDIG_Re=hDIG_Re, hDIG_ALL=hDIG_ALL, SFc=SFc, SFc_Re=SFc_Re, SFc_ALL=SFc_ALL)
        for k, v in dict_masks.items():
            print('{}:'.format(k))
            X = x.loc[v]
            Y = y.loc[v]
            p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(X, Y, interval=interval, step=0.1, clip=2)
            mod_key = 'mod_%s_%s' % (y_key, k)
            mod_key_2sigma = '%s_2sigma' % mod_key
            elines[mod_key] = np.polyval(p, x)
            elines[mod_key_2sigma] = np.polyval(pc, x)
            R_key = 'R_%s' % mod_key
            R_key_2sigma = 'R_%s' % mod_key_2sigma
            elines[R_key] = y - elines[mod_key]
            elines[R_key_2sigma] = y - elines[mod_key_2sigma]
            print(mod_key, mod_key_2sigma, R_key, R_key_2sigma)
            args.props[R_key] = dict(fname=R_key, label=r'${\rm R}_{\rm SB}$', extent=[-4, 1], majloc=5, minloc=2)
            args.props[R_key_2sigma] = dict(fname=R_key_2sigma, label=r'${\rm R}_{\rm SB}$', extent=[-4, 1], majloc=5, minloc=2)
            if k == 'SFc_Re':
                p_SFc = pc
                ax.plot(interval[0:2], np.polyval(p_SFc, interval[0:2]), c='k', ls='--', label='SFG')
                # ax.text(x_bins_center__r[0], np.polyval(p_SFc, x_bins_center__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
            # if k == 'hDIG':
            #     p_hDIG = p
            #     p_hDIG_c = pc
            #     ax.plot(interval[0:2], np.polyval(p_hDIG_c, interval[0:2]), c='k', ls='--', label='RG')
            #     ax.text(x_bins_center__r[0], np.polyval(p_hDIG_c, x_bins_center__r[0]), 'RG', color='k', fontsize=args.fontsize, va='center', ha='right')
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
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)

    x_key = 'log_Mass_gas_Av_gas_rad'
    x = elines[x_key]
    x_label = props[x_key]['label']
    x_extent = props[x_key]['extent']
    x_majloc = props[x_key]['majloc']
    x_minloc = props[x_key]['minloc']
    y_key_list = ['lSFR', 'log_SFR_SF', 'log_SFR_ssp']
    for y_key in y_key_list:
        print('\n##################################')
        k = '%s_%s' % (props[x_key]['fname'], props[y_key]['fname'])
        fname = 'fig_%s' % k
        print('# %s' % fname)
        print('##################################')
        y = elines[y_key]
        y_label = props[y_key]['label']
        y_extent = props[y_key]['extent']
        y_majloc = props[y_key]['majloc']
        y_minloc = props[y_key]['minloc']
        extent = x_extent + y_extent
        prune_x = None
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
        plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                          xlabel=x_label, ylabel=y_label, zlabel=z_label,
                          n_bins_maj_x=x_majloc, n_bins_min_x=x_minloc,
                          n_bins_maj_y=y_majloc, n_bins_min_y=y_minloc,
                          extent=extent, z_extent=z_extent, f=f, ax=ax)
        WHa_Re = elines['EW_Ha_Re']
        SFc_Re = sel_EW_Re & (WHa_Re > args.EW_SF)
        X = x
        Y = y
        # Z = z.loc[SFc_Re]
        # XS, YS, ZS = xyz_clean_sort_interval(X.values, Y.values, Z.values)
        p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(X, Y, interval=extent, step=0.1, clip=2)
        mod_SFR = np.polyval(p, np.array(extent[0:2]) + np.array([0.5, -0.5]))
        ax.plot(np.array(extent[0:2]) + np.array([0.5, -0.5]), mod_SFR, c='k', ls='--')
        # x_bins__r, x_bincenter__r, nbins = create_bins(np.array(extent[0:2]) + np.array([0.5, -0.5]), 0.5)
        # _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, extent)
        # print(y_mean, N_y_mean)
        # ax.plot(x_bincenter__r, y_mean, 'k--')
        # print('p:%.2f s:%.2f' % (pearsonr(X, Y)[0], spearmanr(X, Y)[0]))
        ########################################
        N_AGN_tI_under_bestfit = ((y[mtI] - np.polyval(p, x[mtI])) <= 0).astype('int').sum()
        N_AGN_tII_under_bestfit = ((y[mtII] - np.polyval(p, x[mtII])) <= 0).astype('int').sum()
        N_BFAGN_under_bestfit = ((y[mBFAGN] - np.polyval(p, x[mBFAGN])) <= 0).astype('int').sum()
        N_ALLAGN_under_bestfit = ((y[mALLAGN] - np.polyval(p, x[mALLAGN])) <= 0).astype('int').sum()
        print('# B.F. Type-I AGN under bestfit: %d/%d (%.1f%%)' % (N_AGN_tI_under_bestfit, N_AGN_tI, 100.*N_AGN_tI_under_bestfit/N_AGN_tI))
        print('# B.F. Type-II AGN under bestfit: %d/%d (%.1f%%)' % (N_AGN_tII_under_bestfit, N_AGN_tII, 100.*N_AGN_tII_under_bestfit/N_AGN_tII))
        print('# B.F. AGN under bestfit: %d/%d (%.1f%%)' % (N_BFAGN_under_bestfit, N_BFAGN, 100.*N_BFAGN_under_bestfit/N_BFAGN))
        print('# ALL AGN under bestfit: %d/%d (%.1f%%)' % (N_ALLAGN_under_bestfit, N_ALLAGN, 100.*N_ALLAGN_under_bestfit/N_ALLAGN))
        ########################################
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print('###############################')
        ###############################

    plots_props_list = [
        ['log_Mass_corr', 'log_Mass_gas_Av_gas_rad'],
        ['log_Mass_corr', 'C'],
    ]
    for plot_element in plots_props_list:
        print('\n#############################')
        x_key = plot_element[0]
        y_key = plot_element[1]
        x_key, y_key = 'log_Mass_corr', 'log_Mass_gas_Av_gas_rad'
        k = '%s_%s' % (props[x_key]['fname'], props[y_key]['fname'])
        fname = 'fig_%s' % k
        print('# %s' % fname)
        print('#############################')
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        plot_colored_by_z(elines=elines, args=args, markAGNs=True,
                          x=elines[x_key], y=elines[y_key], z=z,
                          xlabel=props[x_key]['label'], ylabel=props[y_key]['label'], zlabel=z_label,
                          extent=props[x_key]['extent'] + props[y_key]['extent'], z_extent=z_extent,
                          n_bins_maj_x=props[x_key]['majloc'], n_bins_min_x=props[x_key]['minloc'],
                          n_bins_maj_y=props[y_key]['majloc'], n_bins_min_y=props[y_key]['minloc'],
                          prune_x=None, output_name=output_name)
        print('#############################')

    ###################################
    x_key = 'log_Mass_corr'
    x = elines[x_key]
    x_label = props[x_key]['label']
    x_extent = props[x_key]['extent']
    x_majloc = props[x_key]['majloc']
    x_minloc = props[x_key]['minloc']
    ###################################
    for y_key in ['sSFR', 'sSFR_SF', 'sSFR_ssp']:
        k = '%s_%s' % (props[x_key]['fname'], props[y_key]['fname'])
        fname = 'fig_%s' % k
        print('# %s' % fname)
        print('###################################')
        y = elines[y_key]
        y_label = props[y_key]['label']
        y_extent = props[y_key]['extent']
        y_majloc = props[y_key]['majloc']
        y_minloc = props[y_key]['minloc']
        extent = x_extent + y_extent
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        N_rows, N_cols = 1, 1
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
        plot_colored_by_z(elines=elines, args=args, x=x, y=y, z=z, markAGNs=True,
                          xlabel=x_label, ylabel=y_label, zlabel=z_label,
                          z_extent=z_extent, extent=extent,
                          n_bins_maj_x=x_majloc, n_bins_min_x=x_minloc,
                          n_bins_maj_y=y_majloc, n_bins_min_y=y_minloc,
                          prune_x=None, f=f, ax=ax)
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
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print('###################################')
    ###################################

    # #############################
    # ## M-fgas colored by EW_Ha ##
    # #############################
    # print('\n#############################')
    # print('## M-fgas colored by EW_Ha ##')
    # print('#############################')
    # x = elines['log_Mass_corr']
    # y = elines['log_fgas']
    # n_bins_min_x = 5
    # n_bins_maj_y = 5
    # n_bins_min_y = 2
    # output_name = '%s/fig_M_fgas.%s' % (args.figs_dir, args.img_suffix)
    # f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    # N_rows, N_cols = 1, 1
    # # bottom, top, left, right = 0.30, 0.95, 0.2, 0.75
    # bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    # gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    # ax = plt.subplot(gs[0])
    # f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
    #                           ylabel=r'$\log\ f_{\rm gas}$',
    #                           xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
    #                           extent=[8, 12, -5, 0], markAGNs=True,
    #                           n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
    #                           n_bins_maj_x=4, n_bins_min_x=n_bins_min_x, prune_x=prune_x, zlabel=z_label)
    # mtAGN = mtI | mtII
    # WHa = elines['EW_Ha_cen_mean']
    # WHa_Re = elines['EW_Ha_Re']
    # WHa_ALL = elines['EW_Ha_ALL']
    # interval = [9, 11.5, -5, 0]
    # XS, YS, ZS = xyz_clean_sort_interval(x.values, y.values, WHa_Re.values)
    # x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.5)
    # _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, interval)
    # print(x_bincenter__r)
    # # print(y_mean, N_y_mean)
    # # ax.plot(x_bincenter__r, y_mean, 'k--')
    # SFc = ZS > args.EW_SF
    # GV = (ZS > args.EW_hDIG) & (ZS <= args.EW_SF)
    # RG = ZS <= args.EW_hDIG
    # _, _, _, y_mean_SF, N_y_mean_SF, _ = redf_xy_bins_interval(XS[SFc], YS[SFc], x_bins__r, interval)
    # # print(y_mean_SF, N_y_mean_SF)
    # p = np.ma.polyfit(x_bincenter__r, y_mean_SF, 1)
    # ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='k', ls='--', label='SFG')
    # ax.text(x_bincenter__r[0], np.polyval(p, x_bincenter__r[0]), 'SFG', color='k', fontsize=args.fontsize, va='center', ha='right')
    # # _, _, _, y_mean_RG, N_y_mean_RG, _ = redf_xy_bins_interval(XS[RG], YS[RG], x_bins__r, interval)
    # # print(y_mean_RG, N_y_mean_RG)
    # # p = np.ma.polyfit(x_bincenter__r, y_mean_RG, 1)
    # # print(y_mean_RG, p)
    # # ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='r', label='RG')
    # ### above ###
    # N_y_tI = y[mtI].count()
    # N_y_tII = y[mtII].count()
    # # N_y_tIII = y[mtIII].count()
    # N_y_tAGN = y[mtAGN].count()
    # N_y_tI_above = count_y_above_mean(x.loc[mtI].values, y.loc[mtI].values, y_mean, x_bins__r, interval=interval)
    # N_y_tII_above = count_y_above_mean(x.loc[mtII].values, y.loc[mtII].values, y_mean, x_bins__r, interval=interval)
    # # N_y_tIII_above = count_y_above_mean(x.loc[mtIII].values, y.loc[mtIII].values, y_mean, x_bins__r, interval=interval)
    # N_y_tAGN_above = count_y_above_mean(x.loc[mtAGN].values, y.loc[mtAGN].values, y_mean, x_bins__r, interval=interval)
    # print('# B.F. Type-I AGN above mean: %d/%d (%.1f%%)' % (N_y_tI_above, N_y_tI, 100.*N_y_tI_above/N_y_tI))
    # print('# B.F. Type-II AGN above mean: %d/%d (%.1f%%)' % (N_y_tII_above, N_y_tII, 100.*N_y_tII_above/N_y_tII))
    # # print('# 2 crit. AGN above mean: %d/%d (%.1f%%)' % (N_y_tIII_above, N_y_tIII, 100.*N_y_tIII_above/N_y_tIII))
    # print('# ALL AGN above mean: %d/%d (%.1f%%)' % (N_y_tAGN_above, N_y_tAGN, 100.*N_y_tAGN_above/N_y_tAGN))
    # f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    # plt.close(f)
    # print('#############################')
    # ##########################

    # ############################
    # ## M-SFE colored by EW_Ha ##
    # ############################
    # print('\n############################')
    # print('## M-SFE colored by EW_Ha ##')
    # print('############################')
    # x = elines['log_Mass_corr']
    # y = elines['log_SFR_SF'] - elines['log_Mass_gas_Av_gas_rad']
    # n_bins_min_x = 5
    # n_bins_maj_y = 3
    # n_bins_min_y = 4
    # # z = fgas.apply(np.log10)
    # # zlabel = r'$\log\ f_{\rm gas}$'
    # output_name = '%s/fig_M_SFE.%s' % (args.figs_dir, args.img_suffix)
    # f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    # N_rows, N_cols = 1, 1
    # # bottom, top, left, right = 0.30, 0.95, 0.2, 0.75
    # bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    # gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    # ax = plt.subplot(gs[0])
    # sc_kwargs = dict(s=1, vmax=0, vmin=-3, cmap='viridis_r', marker='o', edgecolor='none', alpha=1)
    # f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
    #                           ylabel=r'$\log$ (SFE/yr)',
    #                           xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
    #                           extent=[8, 12, -11, -6], markAGNs=True,
    #                           n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
    #                           n_bins_maj_x=4, n_bins_min_x=n_bins_min_x, prune_x=prune_x, zlabel=z_label) #,
    #                           # sc_kwargs=sc_kwargs, z_maxlocator=3)
    # mtAGN = mtI | mtII
    # WHa = elines['EW_Ha_cen_mean']
    # WHa_Re = elines['EW_Ha_Re']
    # WHa_ALL = elines['EW_Ha_ALL']
    # if interval is None:
    #     interval = [9, 11.5, -5, 0]
    # XS, YS, ZS = xyz_clean_sort_interval(x.values, y.values, WHa_Re.values)
    # x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.5)
    # _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(XS, YS, x_bins__r, interval)
    # # print(y_mean, N_y_mean)
    # # ax.plot(x_bincenter__r, y_mean, 'k--')
    # SFc = ZS > args.EW_SF
    # GV = (ZS > args.EW_hDIG) & (ZS <= args.EW_SF)
    # RG = ZS <= args.EW_hDIG
    # _, _, _, y_mean_SF, N_y_mean_SF, _ = redf_xy_bins_interval(XS[SFc], YS[SFc], x_bins__r, interval)
    # # print(y_mean_SF, N_y_mean_SF)
    # p = np.ma.polyfit(x_bincenter__r, y_mean_SF, 1)
    # ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='k', ls='--', label='SFG')
    # # _, _, _, y_mean_RG, N_y_mean_RG, _ = redf_xy_bins_interval(XS[RG], YS[RG], x_bins__r, interval)
    # # print(y_mean_RG, N_y_mean_RG)
    # # p = np.ma.polyfit(x_bincenter__r, y_mean_RG, 1)
    # # print(y_mean_RG, p)
    # # ax.plot(interval[0:2], np.polyval(p, interval[0:2]), c='r', label='RG')
    # ### above ###
    # N_y_tI = y[mtI].count()
    # N_y_tII = y[mtII].count()
    # # N_y_tIII = y[mtIII].count()
    # N_y_tAGN = y[mtAGN].count()
    # N_y_tI_above = count_y_above_mean(x.loc[mtI].values, y.loc[mtI].values, y_mean, x_bins__r, interval=interval)
    # N_y_tII_above = count_y_above_mean(x.loc[mtII].values, y.loc[mtII].values, y_mean, x_bins__r, interval=interval)
    # # N_y_tIII_above = count_y_above_mean(x.loc[mtIII].values, y.loc[mtIII].values, y_mean, x_bins__r, interval=interval)
    # N_y_tAGN_above = count_y_above_mean(x.loc[mtAGN].values, y.loc[mtAGN].values, y_mean, x_bins__r, interval=interval)
    # print('# B.F. Type-I AGN above mean: %d/%d (%.1f%%)' % (N_y_tI_above, N_y_tI, 100.*N_y_tI_above/N_y_tI))
    # print('# B.F. Type-II AGN above mean: %d/%d (%.1f%%)' % (N_y_tII_above, N_y_tII, 100.*N_y_tII_above/N_y_tII))
    # # print('# 2 crit. AGN above mean: %d/%d (%.1f%%)' % (N_y_tIII_above, N_y_tIII, 100.*N_y_tIII_above/N_y_tIII))
    # print('# ALL AGN above mean: %d/%d (%.1f%%)' % (N_y_tAGN_above, N_y_tAGN, 100.*N_y_tAGN_above/N_y_tAGN))
    # f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    # plt.close(f)
    # print('#############################')
    # ##########################

    ##################################
    ## sSFR_SF-g-r colored by EW_Ha ##
    ##################################
    print('\n##################################')
    print('## sSFR_SF-g-r colored by EW_Ha ##')
    print('##################################')
    x = elines['sSFR_SF']
    y = elines['g_r']
    n_bins_maj_x = 6
    n_bins_min_x = 2
    n_bins_maj_y = 3
    n_bins_min_y = 5
    output_name = '%s/fig_sSFRSF_gr.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
                              xlabel=r'$\log ({\rm sSFR}_{\rm H\alpha}^{\rm SF}/{\rm yr})$',
                              ylabel=r'g-r (mag)',
                              extent=[-14.5, -8.5, 0, 1], markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, zlabel=z_label)
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

    ################################
    ## EWHa-sSFR colored by EW_Ha ##
    ################################
    print('\n################################')
    print('## EWHa-sSFR colored by EW_Ha ##')
    print('################################')
    x = elines['log_EW_Ha_Re']
    y = elines['sSFR']
    extent = [-1, 2.5, -13.5, -8.5]
    n_bins_maj_x = 4
    n_bins_min_x = 2
    n_bins_maj_y = 5
    n_bins_min_y = 2
    output_name = '%s/fig_EWHaRe_sSFR.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
                              xlabel=r'$\log {\rm W}_{{\rm H}\alpha}^{\rm Re}$ (\AA)',
                              ylabel=r'$\log ({\rm sSFR}_{\rm H\alpha}/{\rm yr})$',
                              extent=extent, markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, zlabel=z_label)
    X = x
    Y = y
    p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(X, Y, interval=extent, step=0.1, clip=2)
    borders = np.array([0.2, -0.2])
    mod_sSFR = np.polyval(p, np.array(extent[0:2]) + borders)
    ax.plot(np.array(extent[0:2]) + borders, mod_sSFR, c='k', ls='--')
    N_GV = ((y <= -10.8) & (y > -11.8)).astype('int').sum()
    N_AGN_tI_GV = ((y[mtI] <= -10.8) & (y[mtI] > -11.8)).astype('int').sum()
    N_AGN_tII_GV = ((y[mtII] <= -10.8) & (y[mtII] > -11.8)).astype('int').sum()
    N_BFAGN_GV = ((y[mBFAGN] <= -10.8) & (y[mBFAGN] > -11.8)).astype('int').sum()
    N_ALLAGN_GV = ((y[mALLAGN] <= -10.8) & (y[mALLAGN] > -11.8)).astype('int').sum()
    print('# B.F. Type-I AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tI_GV, N_GV, N_AGN_tI, 100.*N_AGN_tI_GV/N_GV, 100.*N_AGN_tI_GV/N_AGN_tI))
    print('# B.F. Type-II AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tII_GV, N_GV, N_AGN_tII, 100.*N_AGN_tII_GV/N_GV, 100.*N_AGN_tII_GV/N_AGN_tII))
    print('# B.F. AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_BFAGN_GV, N_GV, N_BFAGN, 100.*N_BFAGN_GV/N_GV, 100.*N_BFAGN_GV/N_BFAGN))
    print('# ALL AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_ALLAGN_GV, N_GV, N_ALLAGN, 100.*N_ALLAGN_GV/N_GV, 100.*N_ALLAGN_GV/N_ALLAGN))
    ax.axhline(y=-11.8, c='k', ls='--')
    ax.axhline(y=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('#############################')
    ##########################

    ###################################
    ## EWHa-sSFR_SF colored by EW_Ha ##
    ###################################
    print('\n###################################')
    print('## EWHa-sSFR_SF colored by EW_Ha ##')
    print('###################################')
    x = elines['log_EW_Ha_Re']
    y = elines['sSFR_SF']
    extent = [-1, 2.5, -14.5, -8.5]
    n_bins_maj_x = 4
    n_bins_min_x = 2
    n_bins_maj_y = 3
    n_bins_min_y = 2
    output_name = '%s/fig_EWHaRe_sSFRSF.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=z,
                              xlabel=r'$\log {\rm W}_{{\rm H}\alpha}^{\rm Re}$ (\AA)',
                              ylabel=r'$\log ({\rm sSFR}_{\rm H\alpha}^{\rm SF}/{\rm yr})$',
                              extent=extent, markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, zlabel=z_label)
    X = x
    Y = y
    p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(X, Y, interval=extent, step=0.1, clip=2)
    borders = np.array([0.2, -0.2])
    mod_sSFR = np.polyval(p, np.array(extent[0:2]) + borders)
    ax.plot(np.array(extent[0:2]) + borders, mod_sSFR, c='k', ls='--')
    N_GV = ((y <= -10.8) & (y > -11.8)).astype('int').sum()
    N_AGN_tI_GV = ((y[mtI] <= -10.8) & (y[mtI] > -11.8)).astype('int').sum()
    N_AGN_tII_GV = ((y[mtII] <= -10.8) & (y[mtII] > -11.8)).astype('int').sum()
    N_BFAGN_GV = ((y[mBFAGN] <= -10.8) & (y[mBFAGN] > -11.8)).astype('int').sum()
    N_ALLAGN_GV = ((y[mALLAGN] <= -10.8) & (y[mALLAGN] > -11.8)).astype('int').sum()
    print('# B.F. Type-I AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tI_GV, N_GV, N_AGN_tI, 100.*N_AGN_tI_GV/N_GV, 100.*N_AGN_tI_GV/N_AGN_tI))
    print('# B.F. Type-II AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tII_GV, N_GV, N_AGN_tII, 100.*N_AGN_tII_GV/N_GV, 100.*N_AGN_tII_GV/N_AGN_tII))
    print('# B.F. AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_BFAGN_GV, N_GV, N_BFAGN, 100.*N_BFAGN_GV/N_GV, 100.*N_BFAGN_GV/N_BFAGN))
    print('# ALL AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_ALLAGN_GV, N_GV, N_ALLAGN, 100.*N_ALLAGN_GV/N_GV, 100.*N_ALLAGN_GV/N_ALLAGN))
    ax.axhline(y=-11.8, c='k', ls='--')
    ax.axhline(y=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('#############################')
    ##########################

    ####################################
    ## EWHa-sSFR_ssp colored by EW_Ha ##
    ####################################
    print('\n####################################')
    print('## EWHa-sSFR_ssp colored by EW_Ha ##')
    print('####################################')
    x = elines['log_EW_Ha_Re']
    y = elines['sSFR_ssp']
    extent = [-1, 2.5, -12.5, -8.5]
    n_bins_maj_x = 4
    n_bins_min_x = 2
    n_bins_maj_y = 4
    n_bins_min_y = 2
    output_name = '%s/fig_EWHaRe_sSFRssp.%s' % (args.figs_dir, args.img_suffix)
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    # z = EW_Ha_cen.apply(np.log10)
    # zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
    f, ax = plot_colored_by_z(elines=elines, args=args, f=f, ax=ax, x=x, y=y, z=elines['log_EW_Ha_cen_mean'],
                              xlabel=r'$\log {\rm W}_{{\rm H}\alpha}^{\rm Re}$ (\AA)',
                              ylabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                              extent=extent, markAGNs=True,
                              n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                              n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                              zlabel=props['log_EW_Ha_cen_mean']['label'])
    X = x
    Y = y
    # Z = z.loc[SFc_Re]
    # XS, YS, ZS = xyz_clean_sort_interval(X.values, Y.values, Z.values)
    p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(X, Y, interval=extent, step=0.1, clip=2)
    borders = np.array([0.2, -0.2])
    mod_sSFR = np.polyval(p, np.array(extent[0:2]) + borders)
    ax.plot(np.array(extent[0:2]) + borders, mod_sSFR, c='k', ls='--')
    f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    plt.close(f)
    print('#############################')
    ##########################

    ##############################
    # sSFR vs (NUV_r, g_r, EWHa) #
    ##############################
    for x_key in ['sSFR', 'sSFR_SF', 'sSFR_ssp']:
        print('\n##############################')
        print('# sSFR vs (NUV_r, g_r, EWHa) #')
        print('##############################')
        k = '%s_Salim14' % props[x_key]['fname']
        fname = 'fig_%s' % k
        print('# %s' % fname)
        N_rows, N_cols = 3, 1
        y_key_list = ['g_r', 'NUV_r_SDSS', 'log_EW_Ha_Re']
        f = plot_setup(width=latex_column_width, aspect=N_rows/golden_mean)
        bottom, top, left, right = 0.10, 0.95, 0.15, 0.80
        gs = gridspec.GridSpec(N_rows, N_cols,
                               left=left, bottom=bottom, right=right, top=top,
                               wspace=0., hspace=0.)
        # ax = plt.subplot(gs[0])
        mask = elines[x_key].notna()
        for y_key in y_key_list:
            mask = mask & elines[y_key].notna()
        x = elines.loc[mask, x_key]
        x_label = props[x_key]['label']
        x_majloc = props[x_key]['majloc']
        x_minloc = 5  # props[x_key]['minloc']
        x_extent = props[x_key]['extent']
        row = 0
        for y_key in y_key_list:
            print(y_key)
            y = elines.loc[mask, y_key]
            zm = elines.loc[mask, z_key]
            y_label = props[y_key]['label']
            y_majloc = props[y_key]['majloc']
            y_minloc = props[y_key]['minloc']
            y_extent = props[y_key]['extent']
            extent = x_extent + y_extent
            print(extent, y_key)
            p, pc, XS, YS, x_bins__r, x_bins_center__r, nbins, YS_c__r, N_c__r, sel_c, YS__r, N__r, sel, c_p, c_r, c_p_c, c_r_c = linear_regression_mean(x, y, interval=extent, step=0.1, clip=2)
            mXY = (x.notna() & y.notna())
            print('x:%s:%d  y:%s:%d  all:%d  tIAGN:%d  tIIAGN:%d  AGN:%d' % (x.name, x.notna().sum(), y.name, y.notna().sum(), mXY.sum(), (mXY & mtI).sum(), (mXY & mtII).sum(), (mXY & mBFAGN).sum()))
            ax = plt.subplot(gs[row])
            sc = ax.scatter(x, y, c=zm, cmap='viridis_r', vmax=z_extent[1], vmin=z_extent[0], s=5, marker='o', edgecolor='none', alpha=0.8)
            ax.scatter(x[mtII], y[mtII], **scatter_AGN_tII_kwargs)
            ax.scatter(x[mtI], y[mtI], **scatter_AGN_tI_kwargs)
            ax.set_ylabel(y_label, fontsize=args.fontsize+1)
            ax.set_xlim(extent[0:2])
            ax.set_ylim(extent[2:4])
            tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelleft=True, labelright=False)
            ax.tick_params(**tick_params)
            ax.xaxis.set_major_locator(MaxNLocator(x_majloc, prune=None))
            ax.xaxis.set_minor_locator(AutoMinorLocator(x_minloc))
            ax.yaxis.set_major_locator(MaxNLocator(y_majloc, prune='lower'))
            ax.yaxis.set_minor_locator(AutoMinorLocator(y_minloc))
            ax.axvline(x=-11.8, c='k', ls='--')
            ax.axvline(x=-10.8, c='k', ls='--')
            WHacen = elines['EW_Ha_cen_mean']
            WHaRe = elines['EW_Ha_Re']
            SFG = WHaRe > args.EW_SF
            GVG = (WHaRe > args.EW_hDIG) & (WHaRe <= args.EW_SF)
            RG = WHaRe <= args.EW_hDIG
            masks = dict(SFG=SFG, GVG=GVG, RG=RG)
            colors = dict(SFG='b', GVG='g', RG='r')
            sel = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            xm, ym, zm = ma_mask_xyz(x.loc[sel].values, y.loc[sel].values, z.loc[sel].values)
            rs = runstats(xm.compressed(), ym.compressed(),
                          smooth=True, sigma=1.2,
                          debug=True, gs_prc=True,
                          poly1d=True)
            ax.plot(rs.xS, rs.yS, 'k--', lw=1)
            txt = 'r:%.2f' % rs.Rs
            plot_text_ax(ax, txt, 0.96, 0.95, args.fontsize+2, 'top', 'right', 'k')
            # bins per class
            # for k, v in masks.items():
            #     c = colors[k]
            #     print(k,c)
            #     m = v & sel
            #     xm, ym, zm = ma_mask_xyz(x.loc[m].values, y.loc[m].values, z.loc[m].values)
            #     rs_masked = runstats(xm.compressed(), ym.compressed(),
            #                          smooth=True, sigma=1.2,
            #                          debug=True, gs_prc=True,
            #                          poly1d=True)
            #     print(rs_masked.poly1d_median_slope, rs_masked.poly1d_median_intercept)
            #     ax.plot(rs_masked.xS, rs_masked.yS, color=c, lw=1)
            if y_key == 'log_EW_Ha_Re':
                ax.axhline(y=np.log10(3), c='r', ls='--')
                ax.axhline(y=np.log10(10), c='b', ls='--')
            if 'NUV_r' in y_key:
                ax.axhline(y=4, c='k', ls='--')
                ax.axhline(y=5, c='k', ls='--')
            if args.debug and row == 0:
                plot_text_ax(ax, '%d' % mask.astype('int').sum(), 0.96, 0.95, args.fontsize+2, 'top', 'right', 'k')
            row = row + 1

        cb_width = 0.05
        cb_ax = f.add_axes([right, bottom, cb_width, top-bottom])
        cb = plt.colorbar(sc, cax=cb_ax)
        cb.solids.set(alpha=1)
        cb.set_label(z_label, fontsize=args.fontsize+1)
        cb.locator = MaxNLocator(props[z_key]['majloc'])
        # cb_ax.minorticks_on()
        cb_ax.tick_params(which='both', direction='in')
        cb.update_ticks()

        ax.yaxis.set_major_locator(MaxNLocator(y_majloc, prune=None))
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minloc))
        tick_params['labelbottom'] = True
        ax.tick_params(**tick_params)
        ax.set_xlabel(x_label, fontsize=args.fontsize+1)
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)

    ################################
    ################################
    ## X Y histo colored by EW_Ha ##
    ################################
    ################################
    print('\n################################')
    print('## X Y histo colored by EW_Ha ##')
    print('################################')
    # print(elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean'])
    # print(np.log10(np.abs(elines.loc['2MASXJ01331766+1319567', 'EW_Ha_cen_mean'])))
    m_redshift = (elines['z_stars'] > 1e-6) & (elines['z_stars'] < 0.2)

    # XXX TODO:
    #   Create a process that reads the dimensions (rows X cols) of each list
    # element in order to create subplots with gridspec rows and cols
    plots_props_list = [
        ['log_fgas', 'R_mod_lSFR_SFc_Re'],
        ['SFE', 'R_mod_lSFR_SFc_Re'],
        ['SFE_SF', 'R_mod_lSFR_SFc_Re'],
        ['SFE_ssp', 'R_mod_lSFR_SFc_Re'],
        ['SFE', 'R_mod_lSFR_SFc_Re_2sigma'],
        ['SFE_SF', 'R_mod_lSFR_SFc_Re_2sigma'],
        ['SFE_ssp', 'R_mod_lSFR_SFc_Re_2sigma'],
        ['log_fgas', 'R_mod_log_SFR_SF_SFc_Re'],
        ['SFE', 'R_mod_log_SFR_SF_SFc_Re'],
        ['SFE_SF', 'R_mod_log_SFR_SF_SFc_Re'],
        ['SFE_ssp', 'R_mod_log_SFR_SF_SFc_Re'],
        ['SFE', 'R_mod_log_SFR_SF_SFc_Re_2sigma'],
        ['SFE_SF', 'R_mod_log_SFR_SF_SFc_Re_2sigma'],
        ['SFE_ssp', 'R_mod_log_SFR_SF_SFc_Re_2sigma'],
        ['log_fgas', 'R_mod_log_SFR_ssp_SFc_Re'],
        ['SFE', 'R_mod_log_SFR_ssp_SFc_Re'],
        ['SFE_SF', 'R_mod_log_SFR_ssp_SFc_Re'],
        ['SFE_ssp', 'R_mod_log_SFR_ssp_SFc_Re'],
        ['SFE', 'R_mod_log_SFR_ssp_SFc_Re_2sigma'],
        ['SFE_SF', 'R_mod_log_SFR_ssp_SFc_Re_2sigma'],
        ['SFE_ssp', 'R_mod_log_SFR_ssp_SFc_Re_2sigma'],
        ['Mabs_r', 'g_r'],
        ['Mabs_r_NC', 'g_r_NC'],
        ['log_Mass_corr', 'C'],
        ['log_Mass_corr', 'g_r'],
        ['log_Mass_corr_NC', 'g_r_NC'],
        ['log_Mass_corr', 'lSFR'],
        ['log_Mass_corr_NC', 'lSFR_NC'],
        ['log_Mass_corr', 'log_SFR_SF'],
        ['log_Mass_corr', 'log_SFR_ssp'],
        ['log_Mass_corr', 'log_fgas'],
        ['log_Mass_corr', 'SFE'],
        ['log_Mass_corr', 'SFE_SF'],
        ['log_Mass_corr', 'SFE_ssp'],
        ['log_Mass_corr', 'sSFR'],
        ['log_Mass_corr', 'sSFR_SF'],
        ['log_Mass_corr', 'sSFR_ssp'],
        ['log_Mass_corr', 'ZH_LW_Re_fit'],
        ['log_Mass_corr', 'ZH_MW_Re_fit'],
        ['log_Mass_corr', 'OH_Re_fit_t2'],
        ['log_Mass_corr', 'Age_LW_Re_fit'],
        ['log_Mass_corr', 'Age_MW_Re_fit'],
        ['log_Mass_corr', 'log_age_mean_LW'],
        ['sSFR', 'g_r'],
        ['sSFR_SF', 'g_r'],
        ['sSFR_ssp', 'g_r'],
        ['log_Mass_gas_Av_gas_rad', 'lSFR'],
        ['log_Mass_gas_Av_gas_rad', 'log_SFR_SF'],
        ['log_Mass_gas_Av_gas_rad', 'log_SFR_ssp'],
        # ['sSFR', 'C'],
        # ['sSFR_SF', 'C'],
        # ['sSFR_ssp', 'C'],
        # ['Mabs_i', 'u_i'],
        # ['Mabs_i_NC', 'u_i_NC'],
        # ['Mabs_r', 'u_r'],
        # ['Mabs_r_NC', 'u_r_NC'],
        # ['Mabs_R', 'B_R'],
        # ['Mabs_R_NC', 'B_R_NC'],
        # ['Mabs_V', 'B_V'],
        # ['Mabs_V_NC', 'B_V_NC'],
    ]
    for plot_element in plots_props_list:
        x_key = plot_element[0]
        y_key = plot_element[1]
        x = elines[x_key]
        x_label = props[x_key]['label']
        x_majloc = props[x_key]['majloc']
        x_minloc = props[x_key]['minloc']
        x_extent = props[x_key]['extent']
        prune_x = None
        y = elines[y_key]
        y_label = props[y_key]['label']
        y_majloc = props[y_key]['majloc']
        y_minloc = props[y_key]['minloc']
        y_extent = props[y_key]['extent']
        prune_y = None
        z_key = 'log_EW_Ha_Re'
        # z_key = 'EW_Ha_Re'
        z = elines[z_key]
        z_label = props[z_key]['label']
        z_extent = props[z_key]['extent']
        print('\n################################')
        k = '%s_%s' % (props[x_key]['fname'], props[y_key]['fname'])
        fname = 'fig_histo_%s' % k
        print('# %s' % fname)
        extent = x_extent + y_extent
        aux_mask = m_redshift
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        plot_histo_xy_colored_by_z(elines=elines, args=args, x=x, y=y, z=z,
                                   ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc,
                                   aux_mask=aux_mask,
                                   xlabel=x_label, xrange=extent[0:2],
                                   ylabel=y_label, yrange=extent[2:4],
                                   n_bins_maj_x=x_majloc, n_bins_min_x=x_minloc, prune_x=prune_x,
                                   n_bins_maj_y=y_majloc, n_bins_min_y=y_minloc, prune_y=prune_y,
                                   zlabel=z_label, z_extent=z_extent)
        if 'R_mod' in k:
            ax_sc = plot_RSB(elines, args, x, y, ax_sc, extent)
        if k == 'sSFR_C':
            ax_sc.axvline(x=-11.8, c='k', ls='--')
            ax_sc.axvline(x=-10.8, c='k', ls='--')
        if k == 'M_sSFR':
            ax_sc.axhline(y=-11.8, c='k', ls='--')
            ax_sc.axhline(y=-10.8, c='k', ls='--')
        if k == 'M_OHt2':
            ax_sc = plot_fig_histo_MZR_t2(elines, args, x, y, ax_sc)
        if k == 'M_O3N2':
            ax_sc = plot_fig_histo_MZR_O3N2(elines, args, x, y, ax_sc)
        if k == 'M_ZHMW':
            ax_sc = plot_fig_histo_M_ZHMW(elines, args, x, y, ax_sc)
        if 'M_t' in k:
            ax_sc = plot_fig_histo_M_t(elines, args, x, y, ax_sc)
        if k == 'M_logfgas':
            ax_sc = plot_fig_histo_M_fgas(elines, args, x, y, ax_sc)
        if k == 'M_SFEHa' or k == 'M_SFEHaSF' or k == 'M_SFEssp':
            ax_sc = plot_fig_histo_M_SFE(elines, args, x, y, ax_sc, [9, 11.5, -11, -6])
        # if k[-3::] == '_NC':
        #     if k[0:5] == 'M_SFR':
        #         plot_text_ax(ax_sc, 'NOCEN', 0.04, 0.95, fs+2, 'top', 'left', 'k')
        #     else:
        #         plot_text_ax(ax_sc, 'NOCEN', 0.96, 0.95, fs+2, 'top', 'right', 'k')
        if k == 'M_sSFRHaSF':
            ax_sc.axhline(y=-11.8, c='k', ls='--')
            ax_sc.axhline(y=-10.8, c='k', ls='--')
            N_GV = ((y <= -10.8) & (y > -11.8)).astype('int').sum()
            N_AGN_tI_GV = ((y[mtI] <= -10.8) & (y[mtI] > -11.8)).astype('int').sum()
            N_AGN_tII_GV = ((y[mtII] <= -10.8) & (y[mtII] > -11.8)).astype('int').sum()
            N_BFAGN_GV = ((y[mBFAGN] <= -10.8) & (y[mBFAGN] > -11.8)).astype('int').sum()
            N_ALLAGN_GV = ((y[mALLAGN] <= -10.8) & (y[mALLAGN] > -11.8)).astype('int').sum()
            print('# B.F. Type-I AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tI_GV, N_GV, N_AGN_tI, 100.*N_AGN_tI_GV/N_GV, 100.*N_AGN_tI_GV/N_AGN_tI))
            print('# B.F. Type-II AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_AGN_tII_GV, N_GV, N_AGN_tII, 100.*N_AGN_tII_GV/N_GV, 100.*N_AGN_tII_GV/N_AGN_tII))
            print('# B.F. AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_BFAGN_GV, N_GV, N_BFAGN, 100.*N_BFAGN_GV/N_GV, 100.*N_BFAGN_GV/N_BFAGN))
            print('# ALL AGN GV: %d/%d/%d (%.1f%%/%.1f%%)' % (N_ALLAGN_GV, N_GV, N_ALLAGN, 100.*N_ALLAGN_GV/N_GV, 100.*N_ALLAGN_GV/N_ALLAGN))
        if k == 'M_sSFRssp':
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
        print('################################')
    print('\n################################')

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
    # elines_wmorph['u_i']
    plots_props_list = [
        'log_Mass_corr', 'C', 'Sigma_Mass_cen', 'rat_vel_sigma', 'Re_kpc', 'sSFR', 'sSFR_SF', 'sSFR_ssp', 'bar', 'g_r',
        # 'B_R', 'B_V', 'u_i', 'u_r',
    ]
    for y_key in plots_props_list:
        print('\n############################')
        p = props[y_key]
        k = '%s' % p['fname']
        fname = 'fig_Morph_%s' % k
        print('# %s' % fname)
        y_extent = p['extent']
        y_label = p['label']
        n_bins_maj_y = p['majloc']
        n_bins_min_y = p['minloc']
        y = elines_wmorph[y_key]
        z_key = 'log_EW_Ha_Re'
        # z_key = 'EW_Ha_Re'
        z = elines_wmorph[z_key]
        z_label = props[z_key]['label']
        z_extent = props[z_key]['extent']
        # z = elines_wmorph['EW_Ha_cen_mean']
        # zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        output_name = '%s/%s.%s' % (args.figs_dir, fname, args.img_suffix)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        ax_Hx = plot_x_morph(elines=elines_wmorph, args=args, ax=ax_Hx)
        plot_morph_y_colored_by_z(elines=elines_wmorph, args=args, y=y, z=z, ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, ylabel=y_label, yrange=y_extent, n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=None, zlabel=z_label, z_extent=z_extent)
        # gs.tight_layout(f)
        f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
        plt.close(f)
        print('############################')
    print('\n#################')
    ############################

    # plots_dict = {
    #     'fig_Morph_M': ['log_Mass_corr', [7.5, 12.5], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 6, 2],
    #     'fig_Morph_C': ['C', [0.5, 5.5], r'${\rm R}90/{\rm R}50$', 6, 2],
    #     'fig_Morph_SigmaMassCen': ['Sigma_Mass_cen', [1, 5], r'$\log (\Sigma_\star/{\rm M}_{\odot}/{\rm pc}^2)$ cen', 4, 2],
    #     'fig_Morph_vsigma': ['rat_vel_sigma', [0, 1], r'${\rm V}/\sigma\ ({\rm R} < {\rm Re})$', 2, 5],
    #     'fig_Morph_Re': ['Re_kpc', [0, 25], r'${\rm Re}/{\rm kpc}$', 6, 2],
    #     'fig_Morph_sSFR': ['sSFR', [-13.5, -8.5], r'$\log ({\rm sSFR}_{\rm H\alpha}/{\rm yr})$', 5, 2],
    #     'fig_Morph_sSFRSF': ['sSFR_SF', [-14.5, -8.5], r'$\log ({\rm sSFR}_{\rm H\alpha}^{\rm SF}/{\rm yr})$', 6, 2],
    #     'fig_Morph_sSFRssp': ['sSFR_ssp', [-12.5, -8.5], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 4, 2],
    #     'fig_Morph_bar': ['bar', [-0.2, 2.2], 'bar presence', 3, 1],
    #     'fig_Morph_BR': ['B_R', [0, 1.5], r'B-R (mag)', 3, 5],
    #     'fig_Morph_BV': ['B_V', [0, 1.], r'B-V (mag)', 3, 5],
    #     'fig_Morph_ui': ['u_i', [0, 3.5], r'u-i (mag)', 3, 5],
    #     'fig_Morph_ur': ['u_r', [0, 3.5], r'u-r (mag)', 3, 5],
    #     'fig_Morph_gr': ['g_r', [0, 1], r'g-r (mag)', 3, 5],
    # }
    # for k, v in plots_dict.items():
    #     print('\n############################')
    #     print('# %s' % k)
    #     ykey, yrange, ylabel, n_bins_maj_y, n_bins_min_y = v
    #     y = elines_wmorph[ykey]
    #     z = elines_wmorph['EW_Ha_Re']
    #     zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm Re}$ (\AA)'
    #     # z = elines_wmorph['EW_Ha_cen_mean']
    #     # zlabel = r'$\log {\rm W}_{{\rm H}\alpha}^{\rm cen}$ (\AA)'
    #     f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    #     output_name = '%s/%s.%s' % (args.figs_dir, k, args.img_suffix)
    #     bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    #     gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    #     ax_Hx = plt.subplot(gs[-1, 1:])
    #     ax_Hy = plt.subplot(gs[0:3, 0])
    #     ax_sc = plt.subplot(gs[0:-1, 1:])
    #     ax_Hx = plot_x_morph(elines=elines_wmorph, args=args, ax=ax_Hx)
    #     plot_morph_y_colored_by_z(elines=elines_wmorph, args=args, y=y, z=z.apply(np.log10), ax_Hx=ax_Hx, ax_Hy=ax_Hy, ax_sc=ax_sc, ylabel=ylabel, yrange=yrange, n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=None, zlabel=z_label)
    #     # gs.tight_layout(f)
    #     f.savefig(output_name, dpi=args.dpi, transparent=_transp_choice)
    #     plt.close(f)
    #     print('############################')
    # print('\n#################')
    # ############################

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


######################
### sSFR vs logWHa ###
######################
# x = elines['EW_Ha_cen_mean'].apply(np.log10)
# EW_Ha_Re_SF = 10**(elines['log_SFR_SF'] - elines['lSFR']) * elines['EW_Ha_ALL']
# x = EW_Ha_Re_SF.apply(np.log10)
# xlabel = r'$\log ({\rm W}_{{\rm H}\alpha}^{\rm SF}/{\rm \AA})$'
# plt.xlabel(xlabel, fontsize=20)
# y = elines['log_SFR_ssp'] - elines['log_Mass_corr']
# plt.ylabel(r'$\log ({\rm sSFR_{\rm SSP}}/{\rm yr})$', fontsize=20)
# interval = [-1, 2.5, -12, -9]
# plt.clf()
# plt.scatter(x, y, c='gray', alpha=0.5)
# plt.scatter(x.loc[mtI], y.loc[mtI], marker='*', color='k', s=150)
# plt.scatter(x.loc[mtII], y.loc[mtII], marker='*', color='b', s=150)
# x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.3)
# _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(x.values, y.values, x_bins__r, interval)
# plt.plot(x_bincenter__r, y_mean, 'ro--')
# plt.xlim(interval[0:2])
# plt.ylim(interval[2:4])
#
# x = elines['EW_Ha_cen_mean'].apply(np.log10)
# plt.xlabel(xlabel, fontsize=20)
# y = elines['log_SFR_SF'] - elines['log_Mass_corr']
# plt.ylabel(r'$\log ({\rm sSFR_{{\rm H}\alpha}}/{\rm yr})$', fontsize=20)
# interval = [-1.5, 3, -14.5, -8.5]
# plt.clf()
# plt.scatter(x, y, c='gray', alpha=0.5)
# plt.scatter(x.loc[mtI], y.loc[mtI], marker='*', color='k', s=150)
# plt.scatter(x.loc[mtII], y.loc[mtII], marker='*', color='b', s=150)
# x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.3)
# _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(x.values, y.values, x_bins__r, interval)
# plt.plot(x_bincenter__r, y_mean, 'ro--')
# plt.xlim(interval[0:2])
# plt.ylim(interval[2:4])
#
# x = elines['EW_Ha_cen_mean'].apply(np.log10)
# plt.xlabel(xlabel, fontsize=20)
# y = elines['log_SFR_SF'] - elines['log_SFR_ssp']
# plt.ylabel(r'$\Delta_{\rm SFR}$', fontsize=20)
# interval = [-1.5, 3, -3.5, 1]
# plt.clf()
# plt.scatter(x, y, c='gray', alpha=0.5)
# plt.scatter(x.loc[mtI], y.loc[mtI], marker='*', color='k', s=150)
# plt.scatter(x.loc[mtII], y.loc[mtII], marker='*', color='b', s=150)
# x_bins__r, x_bincenter__r, nbins = create_bins(interval[0:2], 0.3)
# _, _, _, y_mean, N_y_mean, _ = redf_xy_bins_interval(x.values, y.values, x_bins__r, interval)
# plt.plot(x_bincenter__r, y_mean, 'ro--')
# plt.xlim(interval[0:2])
# plt.ylim(interval[2:4])
# plt.axhline(y=0, color='k', ls='-.')
######################

###############################################################################
# END IPYTHON RECIPES #########################################################
###############################################################################
