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


bug = 0.8
EW_SF = 14
EW_hDIG = 3
EW_strong = 6
EW_verystrong = 10
sigma_clip = True
plot = False
# plot = True
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


def fBPT(x, a, b, c):
    return a + (b/(x + c))


def report_ratios(elines):
    groups = [
        ['log_NII_Ha_cen_mean', 'DBName'],
        ['log_SII_Ha_cen_mean', 'DBName'],
        ['log_OI_Ha_cen', 'DBName'],
        ['log_OIII_Hb_cen_mean', 'DBName'],
        ['log_NII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen_mean', 'log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen_mean', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen_mean', 'log_SII_Ha_cen_mean', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
    ]

    for g in groups:
        if 'DBName' in g:
            name = g[0]
        else:
            name = g
        N = elines.groupby(g).ngroups
        print '# %s measured: %d galaxies' % (name, N)


def AGN_candidates(elines):
    print '##################'
    print '# AGN CANDIDATES #'
    print '##################'

    EW_Ha_cen = elines['EW_Ha_cen_mean'].apply(np.abs)
    N_TOT = len(elines)
    g = ['log_NII_Ha_cen_mean', 'log_OIII_Hb_cen_mean']
    N_NO_GAS = N_TOT - elines.groupby(g).ngroups

    L = Lines()
    consts_K01 = L.consts['K01']
    consts_K01_SII_Ha = L.consts['K01_SII_Ha']
    consts_K01_OI_Ha = L.consts['K01_OI_Ha']
    consts_K03 = L.consts['K03']
    consts_S06 = L.consts['S06']
    if sigma_clip:
        consts_K01 = L.sigma_clip_consts['K01']
        consts_K01_SII_Ha = L.sigma_clip_consts['K01_SII_Ha']

    ###############################################################
    sel_NIIHa = ~(elines['log_NII_Ha_cen_mean'].apply(np.isnan))
    sel_OIIIHb = ~(elines['log_OIII_Hb_cen_mean'].apply(np.isnan))
    sel_SIIHa = ~(elines['log_SII_Ha_cen_mean'].apply(np.isnan))
    sel_OIHa = ~(elines['log_OI_Ha_cen'].apply(np.isnan))
    sel_EW = ~(elines['EW_Ha_cen_mean'].apply(np.isnan))
    ###############################################################
    y = elines['log_OIII_Hb_cen_mean']
    # [OIII] vs [NII]
    # AGN/LINER
    x = elines['log_NII_Ha_cen_mean']
    y_mod_K01 = x.apply(fBPT, args=consts_K01)
    y_mod_K03 = x.apply(fBPT, args=consts_K03)
    y_mod_S06 = x.apply(fBPT, args=consts_S06)
    m = sel_NIIHa & sel_OIIIHb & (y > y_mod_K01)
    N_AGN_NII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    # SF
    m = sel_NIIHa & sel_OIIIHb & (y <= y_mod_K01)
    N_SF_K01 = m.values.astype('int').sum()
    N_SF_K01_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    N_SSF_K01 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
    N_VSSF_K01 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
    m = sel_NIIHa & sel_OIIIHb & (y <= y_mod_K03)
    N_SF_K03 = m.values.astype('int').sum()
    N_SF_K03_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)), 'TYPE'] = 6
    N_SSF_K03 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > EW_strong*bug)), 'TYPE'] = 1
    N_VSSF_K03 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
    m = sel_NIIHa & sel_OIIIHb & (y <= y_mod_S06)
    N_SF_S06 = m.values.astype('int').sum()
    N_SF_S06_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    N_SSF_S06 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
    N_VSSF_S06 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII]
    x = elines['log_SII_Ha_cen_mean']
    y_mod_K01_SII = x.apply(fBPT, args=consts_K01_SII_Ha)
    m = sel_NIIHa & sel_OIIIHb & sel_SIIHa & (y > y_mod_K01) & (y > y_mod_K01_SII)
    N_AGN_NII_SII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [OI]
    x = elines['log_OI_Ha_cen']
    y_mod_K01_OI = x.apply(fBPT, args=consts_K01_OI_Ha)
    m = sel_NIIHa & sel_OIIIHb & sel_OIHa & (y > y_mod_K01) & (y > y_mod_K01_OI)
    N_AGN_NII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII] + [OIII] vs [OI]
    m = sel_NIIHa & sel_OIIIHb & sel_SIIHa & sel_OIHa
    m &= (y > y_mod_K01) & (y > y_mod_K01_SII) & (y > y_mod_K01_OI)
    N_AGN_NII_SII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)), 'TYPE'] = 2
    N_SAGN = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > EW_strong*bug)), 'TYPE'] = 3
    N_VSAGN = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
    ###############################################################
    # pAGB
    m = sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen <= EW_hDIG)
    N_pAGB = m.values.astype('int').sum()
    elines.loc[m, 'TYPE'] = 4
    N_pAGB_aboveK01 = (m & (y > y_mod_K01)).values.astype('int').sum()
    N_pAGB_aboveK03 = (m & (y > y_mod_K03)).values.astype('int').sum()
    N_pAGB_aboveS06 = (m & (y > y_mod_S06)).values.astype('int').sum()
    ###############################################################
    N_SF_EW = (sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen > EW_SF*bug)).values.astype('int').sum()
    ###############################################################

    m = (elines['TYPE'] == 2) | (elines['TYPE'] == 3)
    elines.loc[m, 'AGN_FLAG'] = 2
    m = ((elines['SN_broad'] > 8) & ((elines['TYPE'] == 2) | (elines['TYPE'] == 3))) | (elines['broad_by_eye'] == 1)
    elines.loc[m, 'AGN_FLAG'] = 1
    columns_to_csv = [
        'RA', 'DEV', 'log_NII_Ha_cen_mean', 'log_NII_Ha_cen_stddev',
        'log_OIII_Hb_cen_mean', 'log_OIII_Hb_cen_stddev',
        'log_SII_Ha_cen_mean', 'log_SII_Ha_cen_stddev',
        'log_OI_Ha_cen', 'e_log_OI_Ha_cen',
        'EW_Ha_cen_mean', 'EW_Ha_cen_stddev',
        'SN_broad', 'AGN_FLAG'
    ]
    elines.loc[elines['AGN_FLAG'] > 0].to_csv('AGN_CANDIDATES.csv', columns=columns_to_csv)

    print '##################'
    print '# N.TOTAL = %d' % N_TOT
    print '# N.NO GAS = %d' % N_NO_GAS
    print '# N.AGNs/LINERs candidates by [NII]/Ha: %d (%d)' % (N_AGN_NII_Ha, N_AGN_NII_Ha_EW)
    print '# N.AGNs candidates by [NII]/Ha and [SII]/Ha: %d (%d)' % (N_AGN_NII_SII_Ha, N_AGN_NII_SII_Ha_EW)
    print '# N.AGNs candidates by [NII]/Ha and [OI]/Ha: %d (%d)' % (N_AGN_NII_OI_Ha, N_AGN_NII_OI_Ha_EW)
    print '# N.AGNs candidates by [NII]/Ha, [SII]/Ha and [OI]/Ha: %d' % N_AGN_NII_SII_OI_Ha
    print '# N.AGNs candidates (EW>3*%.3f): %d' % (bug, N_AGN)
    print '# N.AGNs strong (EW>%d*%.3f): %d [Very strong (EW>%d*%.3f): %d]' % (EW_strong, bug, N_SAGN, EW_verystrong, bug, N_VSAGN)
    print '# N.P-AGB: %d (above K01: %d - above K03: %d - above S06: %d)' % (N_pAGB, N_pAGB_aboveK01, N_pAGB_aboveK03, N_pAGB_aboveK03)
    print '# N_SF_EW: %d' % N_SF_EW
    print '# N.SF K01: %d (%d S: %d - VS: %d)' % (N_SF_K01, N_SF_K01_EW, N_SSF_K01, N_VSSF_K01)
    print '# N.SF K03: %d (%d S: %d - VS: %d)' % (N_SF_K03, N_SF_K03_EW, N_SSF_K03, N_VSSF_K03)
    print '# N.SF S06: %d (%d S: %d - VS: %d)' % (N_SF_S06, N_SF_S06_EW, N_SSF_S06, N_VSSF_S06)
    print '##################'


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
fnames_long = {
    '3_joint': 'CALIFA_3_joint_classnum.pandas.csv',
    'basic_joint': 'CALIFA_basic_joint.pandas.csv',
    'cen_broad': 'get_CALIFA_cen_broad.pandas.csv',
    'mag_cubes_v2.2': 'get_mag_cubes_v2.2.pandas.csv',
    'RA_DEC': 'get_RA_DEC.pandas.csv',
    'elines': 'get_proc_elines_CALIFA.clean.pandas.csv',
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
                                    index_col=False,
                                    )
    df[key_dataframe].set_index('DBName', inplace=True, drop=False)

# TODO: What I need?
# Morpholgy, Stellar Mass, SFR, v, sigma, Ha, Hb, N2, O3, S2, O1, R90, R50,
# EWHa, u, g, r, i, z, Mu, Mg, Mr, Mi, Mz, Stellar Age (Lum and Mass weighted),
# Stellar and Gas Metallicity, Gas Mass

###############################################################################
# Setting some initial valyes #################################################
###############################################################################
# BROAD BY EYE
df['elines']['broad_by_eye'] = 0
with open('%s/list_Broad_by_eye.pandas.csv' % csv_dir, 'r') as f:
    for l in f.readlines():
        DBName = l.strip()
        if DBName in df['elines'].index:
            df['elines'].loc[DBName, 'broad_by_eye'] = 1
        else:
            print '%s: not in %s' % (DBName, fnames_long['elines'])
        print '%s: %d' % (DBName, df['elines'].loc[DBName, 'broad_by_eye'])
df['elines']['broad'] = 0
df['elines']['MORPH'] = 'none'
df['elines']['morph'] = -1
df['elines']['SN_broad'] = df['cen_broad']['Nsigma']
df['elines'].loc[df['elines']['SN_broad'] <= 0, 'SN_broad'] = 0.
df['elines']['C'] = df['mag_cubes_v2.2']['C']
df['elines']['e_C'] = df['mag_cubes_v2.2']['error_C']
df['elines']['Mabs_R'] = df['mag_cubes_v2.2']['R_band_mag']
df['elines']['e_Mabs_R'] = df['mag_cubes_v2.2']['R_band_mag_error']
df['elines']['B_V'] = df['mag_cubes_v2.2']['B_V']
df['elines']['e_B_V'] = df['mag_cubes_v2.2']['error_B_V']
df['elines']['B_R'] = df['mag_cubes_v2.2']['B_R']
df['elines']['e_B_R'] = df['mag_cubes_v2.2']['error_B_R']
df['elines']['morph'] = df['3_joint']['hubtyp']
df['elines']['RA'] = df['basic_joint']['ra']
df['elines']['DEC'] = df['basic_joint']['de']
df['elines']['RA'] = df['RA_DEC']['RA']
df['elines']['DEC'] = df['RA_DEC']['DEC']
df['elines']['bar'] = df['3_joint']['bar']
df['elines']['TYPE'] = 0
df['elines']['AGN_FLAG'] = 0
###############################################################################

###############################################################################
report_ratios(df['elines'])
###############################################################################

###############################################################################
AGN_candidates(df['elines'])
###############################################################################

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

    sel = create_selections_BPT(df['elines'], bug=bug, EW_strong=EW_strong, EW_verystrong=EW_verystrong)
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
    ax.scatter(x[sel['AGN_NII_SII_OI_Ha_EW']], y[sel['AGN_NII_SII_OI_Ha_EW']], **scatter_AGN_kwargs)
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
    ax.scatter(x[sel['AGN_NII_SII_OI_Ha_EW']], y[sel['AGN_NII_SII_OI_Ha_EW']], **scatter_AGN_kwargs)
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
    ax.scatter(x[sel['AGN_NII_SII_OI_Ha_EW']], y[sel['AGN_NII_SII_OI_Ha_EW']], **scatter_AGN_kwargs)
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
