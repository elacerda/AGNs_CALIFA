import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, \
                              ScalarFormatter
from pytu.plots import add_subplot_axes, plot_text_ax


# plot = False
plot = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)
cmap_R = plt.cm.copper
minorLocator = AutoMinorLocator(5)
_transp_choice = False
_dpi_choice = 300
img_suffix = 'pdf'


def AGN_candidates(elines, sigma_clip=False):
    print '##################'
    print '# AGN CANDIDATES #'
    print '##################'

    log_NII_Ha_cen = elines['log_NII_Ha_cen_mean']
    log_SII_Ha_cen = elines['log_SII_Ha_cen_mean']
    log_OI_Ha_cen = elines['log_OI_Ha_cen']
    log_OIII_Hb_cen = elines['log_OIII_Hb_cen_mean']
    EW_Ha_cen = elines['EW_Ha_cen_mean']

    sel = create_AGN_selections(elines)

    groups = [
        ['log_NII_Ha_cen_mean', 'DBName'],
        ['log_SII_Ha_cen_mean', 'DBName'],
        ['log_OI_Ha_cen', 'DBName'],
        ['log_OIII_Hb_cen_mean', 'DBName'],
        ['log_NII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen_mean', 'log_SII_Ha_cen_mean', 'log_OI_Ha_cen',
         'log_OIII_Hb_cen_mean'],
    ]

    for g in groups:
        if 'DBName' in g:
            name = g[0]
        else:
            name = g
        N = elines.groupby(g).ngroups
        print '# %s measured: %d galaxies' % (name, N)

    m = sel['AGN_NII_Ha']
    N_AGN_NII_Ha = elines['DBName'][m].count()
    m = sel['AGN_NII_SII_Ha']
    N_AGN_NII_SII_Ha = elines['DBName'][m].count()
    m = sel['AGNTypeOne_NII_SII_Ha']
    N_AGNTypeOne_NII_SII_Ha = elines['DBName'][m].count()
    m = sel['AGN_NII_OI_Ha']
    N_AGN_NII_OI_Ha = elines['DBName'][m].count()
    m = sel['AGNTypeOne_NII_OI_Ha']
    N_AGNTypeOne_NII_OI_Ha = elines['DBName'][m].count()

    print '##################'
    print '# AGNs/LINERs classified by [NII]/Ha: %d' % N_AGN_NII_Ha
    print '# AGNs classified by [NII]/Ha and [SII]/Ha: %d' % N_AGN_NII_SII_Ha
    print '# AGNs classified by [NII]/Ha and [OI]/Ha: %d' % N_AGN_NII_OI_Ha
    # print '# TYPE-1 AGNs classified by [NII]/Ha and [SII]/Ha: %d' % N_AGNTypeOne_NII_SII_Ha
    # print '# TYPE-1 AGNs classified by [NII]/Ha and [OI]/Ha: %d' % N_AGNTypeOne_NII_OI_Ha
    print '##################'

def create_AGN_selections(elines, bug=1):
    L = Lines()

    log_NII_Ha_cen = elines['log_NII_Ha_cen_mean']
    log_SII_Ha_cen = elines['log_SII_Ha_cen_mean']
    log_OI_Ha_cen = elines['log_OI_Ha_cen']
    log_OIII_Hb_cen = elines['log_OIII_Hb_cen_mean']
    EW_Ha_cen = -1. * elines['EW_Ha_cen_mean']

    aboveK01NIIHa = ~(L.belowlinebpt('K01', log_NII_Ha_cen, log_OIII_Hb_cen))
    aboveK01SIIHa = ~(L.belowlinebpt('K01_SII_Ha', log_SII_Ha_cen,
                                     log_OIII_Hb_cen))
    aboveK06SIIHa = ~(L.belowlinebpt('K06_SII_Ha', log_SII_Ha_cen,
                                     log_OIII_Hb_cen))
    aboveK01OIHa = ~(L.belowlinebpt('K01_OI_Ha', log_OI_Ha_cen,
                                    log_OIII_Hb_cen))
    aboveK06OIHa = ~(L.belowlinebpt('K06_OI_Ha', log_OI_Ha_cen,
                                    log_OIII_Hb_cen))
    sel_no_hDIG = EW_Ha_cen > 3*bug
    sel_AGNTypeOne_EW = EW_Ha_cen > 6*bug
    sel_SAGN_EW = EW_Ha_cen > 10*bug
    sel_AGN_NII_Ha = aboveK01NIIHa
    sel_AGN_SII_Ha = aboveK01SIIHa & aboveK06SIIHa
    sel_AGN_OI_Ha = aboveK01OIHa & aboveK06OIHa

    sel = {
        'no_hDIG': sel_no_hDIG,
        'AGNTypeOne_EW': sel_AGNTypeOne_EW,
        'SAGN_EW': sel_SAGN_EW,
        'AGN_NII_Ha': sel_AGN_NII_Ha,
        'AGN_SII_Ha': sel_AGN_SII_Ha,
        'AGN_OI_Ha': sel_AGN_OI_Ha,
        'AGN_NII_Ha_EW': sel_no_hDIG & sel_AGN_NII_Ha,
        'AGN_NII_SII_Ha': sel_no_hDIG & sel_AGN_NII_Ha & sel_AGN_SII_Ha,
        'AGN_NII_OI_Ha': sel_no_hDIG & sel_AGN_NII_Ha & sel_AGN_OI_Ha,
        'AGNTypeOne_NII_SII_Ha': sel_AGNTypeOne_EW & sel_AGN_NII_Ha & sel_AGN_SII_Ha,
        'AGNTypeOne_NII_OI_Ha': sel_AGNTypeOne_EW & sel_AGN_NII_Ha & sel_AGN_OI_Ha,
        'SAGN_NII_SII_Ha': sel_SAGN_EW & sel_AGN_NII_Ha & sel_AGN_SII_Ha,
        'SAGN_NII_OI_Ha': sel_SAGN_EW & sel_AGN_NII_Ha & sel_AGN_OI_Ha,
    }

    return sel


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


dflt_kw_scatter = dict(marker='o', edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, frac=0.07, debug=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')

# morphology
morph_name = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'S0', 'S0a', 'Sa', 'Sab',
              'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'I', 'BCD']
# morph_name=['E0','E1','E2','E3','E4','E5','E6','E7','S0','S0a','Sa','Sab','Sb',
#            'Sbc','Sc','Scd','Sd','Sdm','I','BCD']
# morph_name = {'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6,
#               'E7': 7, 'S0': 8, 'S0a': 9, 'Sa': 10, 'Sab': 11, 'Sb': 12,
#               'Sbc': 13, 'Sc': 14, 'Scd': 15, 'Sd': 16, 'Sdm': 17, 'I': 18,
#               'BCD': 19}

# Files and directories
csv_dir = 'csv'
fname1 = 'CALIFA_3_joint_classnum.pandas.csv'
fname2 = 'CALIFA_basic_joint.pandas.csv'
fname3 = 'get_CALIFA_cen_broad.pandas.csv'
fname4 = 'get_mag_cubes_v2.2.pandas.csv'
fname5 = 'get_RA_DEC.pandas.csv'
fname6 = 'get_proc_elines_CALIFA.clean.pandas.csv'
# skip_header_nlines = {
#     'CALIFA_3_joint_classnum.pandas.csv': 14,
#     'CALIFA_basic_joint.pandas.csv': 18,
#     'get_CALIFA_cen_broad.pandas.csv': 13,
#     'get_mag_cubes_v2.2.pandas.csv': 63,
#     'get_RA_DEC.pandas.csv': 0,
#     'get_proc_elines_CALIFA.clean.pandas.csv': 521,
# }
fnames_short = {
    'CALIFA_3_joint_classnum.pandas.csv': '3_joint',
    'CALIFA_basic_joint.pandas.csv': 'basic_joint',
    'get_CALIFA_cen_broad.pandas.csv': 'cen_broad',
    'get_mag_cubes_v2.2.pandas.csv': 'mag_cubes_v2.2',
    'get_RA_DEC.pandas.csv': 'RA_DEC',
    'get_proc_elines_CALIFA.clean.pandas.csv': 'elines',
}

# Read CSV files
df = {}
na_values = ['BAD', 'nan', -999, '-inf', 'inf']
for k, v in fnames_short.iteritems():
    f_path = '%s/%s' % (csv_dir, k)
    key_dataframe = fnames_short[k]
    df[key_dataframe] = pd.read_csv(f_path,
                                    na_values=na_values,
                                    sep=',',
                                    comment='#',
                                    header='infer',
                                    # skiprows=skip_header_nlines[k],
                                    )

# What I need?
# Morpholgy, Stellar Mass, SFR, v, sigma, Ha, Hb, N2, O3, S2, O1, R90, R50,
# EWHa, u, g, r, i, z, Mu, Mg, Mr, Mi, Mz, Stellar Age (Lum and Mass weighted),
# Stellar and Gas Metallicity, Gas Mass

# AGN candidates
AGN_candidates(df['elines'])

if plot:
    L = Lines()
    log_NII_Ha_cen = df['elines']['log_NII_Ha_cen_mean']
    elog_NII_Ha_cen = df['elines']['log_NII_Ha_cen_stddev']
    log_SII_Ha_cen = df['elines']['log_SII_Ha_cen_mean']
    elog_SII_Ha_cen = df['elines']['log_SII_Ha_cen_stddev']
    log_OI_Ha_cen = df['elines']['log_OI_Ha_cen']
    elog_OI_Ha_cen = df['elines']['e_log_OI_Ha_cen']
    log_OIII_Hb_cen = df['elines']['log_OIII_Hb_cen_mean']
    elog_OIII_Hb_cen = df['elines']['log_OIII_Hb_cen_stddev']
    EW_Ha_cen = -1. * df['elines']['EW_Ha_cen_mean']
    eEW_Ha_cen = df['elines']['EW_Ha_cen_stddev']

    sel = create_AGN_selections(df['elines'])
    ##########################
    ##########################
    ### BPT colored by EW_Ha
    ##########################
    f = plot_setup(width=latex_text_width, aspect=1/3.)
    N_rows, N_cols = 1, 3
    bottom, top, left, right = 0.18, 0.95, 0.08, 0.9
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right,
                           top=top, wspace=0., hspace=0.)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    fs = 7
    y = log_OIII_Hb_cen
    z = EW_Ha_cen.abs().apply(np.log10)
    scatter_kwargs = dict(c=z, s=2, vmax=2.5, vmin=-1, cmap='viridis', marker='o',
                          edgecolor='none')
    scatter_AGN_kwargs = dict(s=50, linewidth=0.01, marker='*', facecolor='none',
                              edgecolor='blue')
    scatter_AGNTypeOne_kwargs = dict(s=50, linewidth=0.01, marker='*', facecolor='none',
                                     edgecolor='k')
    ##########################
    ### NII/Ha
    ax = ax0
    x = log_NII_Ha_cen
    extent = [-1.6, 1, -1.2, 1.5]
    sc = ax.scatter(x, y, **scatter_kwargs)
    ax.scatter(x[sel['AGN_NII_OI_Ha']], y[sel['AGN_NII_OI_Ha']], **scatter_AGN_kwargs)
    ax.scatter(x[sel['AGNTypeOne_NII_OI_Ha']], y[sel['AGNTypeOne_NII_OI_Ha']], **scatter_AGNTypeOne_kwargs)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='both', which='both', direction='in',
                   bottom=True, top=True, left=True, right=True,
                   labelbottom='on', labeltop='off',
                   labelleft='off', labelright='off')
    ax.plot(L.x['K01'], L.y['K01'], 'k--')
    ax.plot(L.x['S06'], L.y['S06'], 'k-.')
    ax.plot(L.x['K03'], L.y['K03'], 'k-')
    ax.set_xlabel(r'$\log\ ({\rm [NII]}/{\rm H\alpha})$', fontsize=fs+4)
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True,
                   left=True, right=True, labelbottom='on', labeltop='off',
                   labelleft='on', labelright='off')
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN/LINER', 0.9, 0.95, fs+2, 'top', 'right', 'k')
    ##########################
    # SII/Ha
    ##########################
    ax = ax1
    x = log_SII_Ha_cen
    sc = ax.scatter(x, y, **scatter_kwargs)
    ax.scatter(x[sel['AGN_NII_OI_Ha']], y[sel['AGN_NII_OI_Ha']], **scatter_AGN_kwargs)
    ax.scatter(x[sel['AGNTypeOne_NII_OI_Ha']], y[sel['AGNTypeOne_NII_OI_Ha']], **scatter_AGNTypeOne_kwargs)
    extent = [-1.6, 1, -1.2, 1.5]
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(r'$\log\ ({\rm [SII]}/{\rm H\alpha})$', fontsize=fs+4)
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True,
                   left=True, right=True, labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    ax.plot(L.x['K01_SII_Ha'], L.y['K01_SII_Ha'], 'k--')
    ax.plot(L.x['K06_SII_Ha'], L.y['K06_SII_Ha'], 'k-.')
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN', 0.65, 0.95, fs+2, 'top', 'right', 'k')
    plot_text_ax(ax, 'LINER', 0.95, 0.85, fs+2, 'top', 'right', 'k')
    ##########################
    # OI/Ha
    ##########################
    ax = ax2
    x = log_OI_Ha_cen
    extent = [-3, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, **scatter_kwargs)
    ax.scatter(x[sel['AGN_NII_OI_Ha']], y[sel['AGN_NII_OI_Ha']], **scatter_AGN_kwargs)
    ax.scatter(x[sel['AGNTypeOne_NII_OI_Ha']], y[sel['AGNTypeOne_NII_OI_Ha']], **scatter_AGNTypeOne_kwargs)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', fontsize=fs+4)
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True,
                   left=True, right=True, labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    cb_ax = f.add_axes([right, bottom, 0.02, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'$\log\ |{\rm EW(H\alpha)}|$', fontsize=fs+4)
    ax.plot(L.x['K01_OI_Ha'], L.y['K01_OI_Ha'], 'k--')
    ax.plot(L.x['K06_OI_Ha'], L.y['K06_OI_Ha'], 'k-.')
    plot_text_ax(ax, 'SF', 0.1, 0.05, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'AGN', 0.65, 0.95, fs+2, 'top', 'right', 'k')
    plot_text_ax(ax, 'LINER', 0.95, 0.85, fs+2, 'top', 'right', 'k')
    ##########################
    f.text(0.01, 0.5, r'$\log\ ({\rm [OIII]}/{\rm H\beta})$', va='center',
           rotation='vertical', fontsize=fs+4)
    f.savefig('fig1.%s' % img_suffix, dpi=_dpi_choice, transparent=_transp_choice)
