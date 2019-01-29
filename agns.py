#!/home/lacerda/anaconda2/bin/python
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytu.plots import add_subplot_axes, plot_text_ax, plot_scatter_histo
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, ScalarFormatter


bug = 0.8
EW_SF = 14
EW_hDIG = 3
EW_strong = 6
EW_verystrong = 10
sigma_clip = True
plot = False
plot = True
# print_color = True
print_color = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['legend.numpoints'] = 1
_transp_choice = False
_dpi_choice = 300
img_suffix = 'pdf'
verbose = 'vv'
fs = 6


if print_color:
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        B = '\033[1m'
        UNDERLINE = '\033[4m'
        E = '\033[0m'
else:
    class color:
        PURPLE = ''
        CYAN = ''
        DARKCYAN = ''
        BLUE = ''
        GREEN = ''
        YELLOW = ''
        RED = ''
        B = ''
        UNDERLINE = ''
        E = ''


def fBPT(x, a, b, c):
    return a + (b/(x + c))


# morphology
morph_name = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'I', 'BCD']
# morph_name = {'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6,
#               'E7': 7, 'S0': 8, 'S0a': 9, 'Sa': 10, 'Sab': 11, 'Sb': 12,
#               'Sbc': 13, 'Sc': 14, 'Scd': 15, 'Sd': 16, 'Sdm': 17, 'I': 18,
#               'BCD': 19}
# morph_name = ['E0','E1','E2','E3','E4','E5','E6','E7','S0','S0a','Sa','Sab','Sb', 'Sbc','Sc','Scd','Sd','Sdm','I','BCD']

# Files and directories
csv_dir = 'csv'
fname1 = 'CALIFA_3_joint_classnum.pandas.csv'
fname2 = 'CALIFA_basic_joint.pandas.csv'
fname3 = 'get_CALIFA_cen_broad.pandas.csv'
fname4 = 'get_mag_cubes_v2.2.pandas.csv'
fname5 = 'get_RA_DEC.pandas.csv'
fname6 = 'get_proc_elines_CALIFA.clean.pandas.csv'
fname6 = 'NII_Ha_fit.csv'
fnames_short = {
    'CALIFA_3_joint_classnum.pandas.csv': '3_joint',
    'CALIFA_basic_joint.pandas.csv': 'basic_joint',
    'get_CALIFA_cen_broad.pandas.csv': 'cen_broad',
    'get_mag_cubes_v2.2.pandas.csv': 'mag_cubes_v2.2',
    'get_RA_DEC.pandas.csv': 'RA_DEC',
    'get_proc_elines_CALIFA.clean.pandas.csv': 'elines',
    'NII_Ha_fit.csv': 'broad_fit',
}
fnames_long = {
    '3_joint': 'CALIFA_3_joint_classnum.pandas.csv',
    'basic_joint': 'CALIFA_basic_joint.pandas.csv',
    'cen_broad': 'get_CALIFA_cen_broad.pandas.csv',
    'mag_cubes_v2.2': 'get_mag_cubes_v2.2.pandas.csv',
    'RA_DEC': 'get_RA_DEC.pandas.csv',
    'elines': 'get_proc_elines_CALIFA.clean.pandas.csv',
    'broad_fit': 'NII_Ha_fit.csv',
}
# Read CSV files
df = {}
na_values = ['BAD', 'nan', -999, '-inf', 'inf']
for k, v in fnames_short.iteritems():
    f_path = '%s/%s' % (csv_dir, k)
    # print f_path
    key_dataframe = fnames_short[k]
    df[key_dataframe] = pd.read_csv(f_path, na_values=na_values, sep=',', comment='#', header='infer', index_col=False)
    df[key_dataframe].set_index('DBName', inplace=True, drop=False)

# TODO: What I need?
# Morpholgy, Stellar Mass, SFR, v, sigma, Ha, Hb, N2, O3, S2, O1, R90, R50,
# EWHa, u, g, r, i, z, Mu, Mg, Mr, Mi, Mz, Stellar Age (Lum and Mass weighted),
# Stellar and Gas Metallicity, Gas Mass

###############################################################################
# SETTING INITIAL VALUES ######################################################
###############################################################################
# BROAD BY EYE
df['elines']['broad_by_eye'] = 0
with open('%s/list_Broad_by_eye.pandas.csv' % csv_dir, 'r') as f:
    for l in f.readlines():
        if l[0] != '#':
            DBName = l.strip()
            if DBName in df['elines'].index:
                df['elines'].loc[DBName, 'broad_by_eye'] = 1
                print '%s%s%s: broad-line by eye' % (color.B, DBName, color.E)
            else:
                print '%s: not in %s' % (DBName, fnames_long['elines'])

# Populating dataframe elines joining different data from other dataframes
df['elines']['C'] = df['mag_cubes_v2.2']['C']
df['elines']['e_C'] = df['mag_cubes_v2.2']['error_C']
df['elines']['Mabs_i'] = df['mag_cubes_v2.2']['i_band_abs_mag']
df['elines']['e_Mabs_i'] = df['mag_cubes_v2.2']['i_band_abs_mag_error']
df['elines']['Mabs_R'] = df['mag_cubes_v2.2']['R_band_abs_mag']
df['elines']['e_Mabs_R'] = df['mag_cubes_v2.2']['R_band_abs_mag_error']
df['elines']['B_V'] = df['mag_cubes_v2.2']['B_V']
df['elines']['e_B_V'] = df['mag_cubes_v2.2']['error_B_V']
df['elines']['B_R'] = df['mag_cubes_v2.2']['B_R']
df['elines']['e_B_R'] = df['mag_cubes_v2.2']['error_B_R']
df['elines']['u'] = df['mag_cubes_v2.2']['u_band_mag']
df['elines']['i'] = df['mag_cubes_v2.2']['i_band_mag']
df['elines']['redshift'] = df['mag_cubes_v2.2']['redshift']
df['elines']['morph'] = df['3_joint']['hubtyp']
df['elines']['RA'] = df['basic_joint']['ra']
df['elines']['DEC'] = df['basic_joint']['de']
df['elines']['RA'] = df['RA_DEC']['RA']
df['elines']['DEC'] = df['RA_DEC']['DEC']
df['elines']['bar'] = df['3_joint']['bar']
df['elines']['SN_broad'] = df['cen_broad']['Nsigma']
df['elines']['broad'] = 0
df['elines']['TYPE'] = 0
df['elines']['AGN_FLAG'] = 0
df['elines']['MORPH'] = 'none'
df['elines']['GalaxyName'] = df['3_joint']['REALNAME']
df['elines']['GalaxyName'] = df['elines']['GalaxyName'].fillna('')
df['elines']['morph'] = df['elines']['morph'].fillna(-1)
df['elines']['Ha_broad'] = df['broad_fit']['Ha_broad']
df['elines']['Ha_narrow'] = df['broad_fit']['Ha_narrow']
df['elines']['NII_6583'] = df['broad_fit']['NII_6583']
df['elines']['NII_6548'] = df['broad_fit']['NII_6548']
df['elines'].loc[df['elines']['SN_broad'] <= 0, 'SN_broad'] = 0.
df['elines'].loc[df['elines']['log_Mass'] < 0, 'log_Mass'] = np.nan
df['elines'].loc[df['elines']['lSFR'] < -10, 'lSFR'] = np.nan
df['elines'].loc[df['elines']['lSFR_NO_CEN'] < -10, 'lSFR_NO_CEN'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas'] == -12, 'log_Mass_gas'] = np.nan
df['elines']['log_NII_Ha_cen_fit'] = np.log10(df['elines']['NII_6583'] / df['elines']['Ha_narrow'])
# f = np.log10(df['elines']['F_Ha_cen']) - np.log10(df['elines']['Ha_narrow'])  # -16 +16
# df['elines']['log_SII_Ha_cen_fit'] = df['elines']['log_SII_Ha_cen_mean'] + f
# df['elines']['log_OI_Ha_cen_fit'] = df['elines']['log_OI_Ha_cen']  + f

#####################
# check unique names
#####################
# gal_indexes = []
# for k in df.iterkeys():
#     x = set(df[k].index)
#     gal_indexes = list(set(gal_indexes) & x) + list(set(gal_indexes) ^ x)
# for n in sorted(gal_indexes):
#     print n
# sys.exit()

##########################
# MySQL dump for morph DB
##########################
# +------------+-------------+------+-----+---------+-------+
# | Field      | Type        | Null | Key | Default | Extra |
# +------------+-------------+------+-----+---------+-------+
# | GalID      | int(4)      | NO   | PRI | NULL    |       |
# | DBName     | varchar(50) | NO   |     | NULL    |       |
# | GalaxyName | varchar(50) | YES  |     | NULL    |       |
# | RA         | double      | YES  |     | NULL    |       |
# | DECL       | double      | YES  |     | NULL    |       |
# | comment    | blob        | YES  |     | NULL    |       |
# | WarningID  | int(2)      | YES  | MUL | NULL    |       |
# | MorphID    | int(2)      | YES  | MUL | NULL    |       |
# +------------+-------------+------+-----+---------+-------+
###########################################################
# print to create database to morphological classification
###########################################################
# k = 0
# print 'INSERT INTO morph_class.to_classify VALUES'
# for i, reg in df['elines'].iterrows():
#     print "(%d, '%s', '%s', '%f', '%f', '', -1, %d)," % (k, i, reg['GalaxyName'], reg['RA'], reg['DEC'], int(reg['morph']))
#     k = k + 1
# print ';'
# sys.exit()

# REMOVE SOME GALS FROM AGN STUDY
with open('%s/remove_gals_AGNpaper.csv' % csv_dir, 'r') as f:
    DBNames_to_drop = []
    for l in f.readlines():
        if l[0] != '#':
            DBName = l.strip()
            if DBName in df['elines'].index:
                DBNames_to_drop.append(DBName)
                print '%s%s%s: removing galaxy from analisys' % (color.B, DBName, color.E)
            else:
                print '%s: not in %s' % (DBName, fnames_long['elines'])
    if len(DBNames_to_drop) > 0:
        df_elines_clean = df['elines'].drop(DBNames_to_drop, axis=0)

elines = df_elines_clean.copy()
del df_elines_clean;

# log_NII_Ha_cen = elines['log_NII_Ha_cen_mean']
log_NII_Ha_cen = elines['log_NII_Ha_cen_fit']
elog_NII_Ha_cen = elines['log_NII_Ha_cen_stddev']
log_SII_Ha_cen = elines['log_SII_Ha_cen_mean']
elog_SII_Ha_cen = elines['log_SII_Ha_cen_stddev']
log_OI_Ha_cen = elines['log_OI_Ha_cen']
elog_OI_Ha_cen = elines['e_log_OI_Ha_cen']
log_OIII_Hb_cen = elines['log_OIII_Hb_cen_mean']
elog_OIII_Hb_cen = elines['log_OIII_Hb_cen_stddev']
EW_Ha_cen = elines['EW_Ha_cen_mean'].apply(np.abs)
eEW_Ha_cen = elines['EW_Ha_cen_stddev']
###############################################################
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
###############################################################
# [OIII] vs [NII]
###############################################################
# AGN/LINER
y_mod_K01 = log_NII_Ha_cen.apply(fBPT, args=consts_K01)
y_mod_K03 = log_NII_Ha_cen.apply(fBPT, args=consts_K03)
y_mod_S06 = log_NII_Ha_cen.apply(fBPT, args=consts_S06)
###############################################################
###############################################################
# [OIII] vs [NII] + [OIII] vs [SII]
###############################################################
# AGN
y_mod_K01_SII = log_SII_Ha_cen.apply(fBPT, args=consts_K01_SII_Ha)
###############################################################
###############################################################
# [OIII] vs [NII] + [OIII] vs [OI]
###############################################################
y_mod_K01_OI = log_OI_Ha_cen.apply(fBPT, args=consts_K01_OI_Ha)
###############################################################
###############################################################
# SELECTIONS
###############################################################
sel_NIIHa = ~(log_NII_Ha_cen.apply(np.isnan))
sel_OIIIHb = ~(log_OIII_Hb_cen.apply(np.isnan))
sel_SIIHa = ~(log_SII_Ha_cen.apply(np.isnan))
sel_OIHa = ~(log_OI_Ha_cen.apply(np.isnan))
sel_EW = ~(EW_Ha_cen.apply(np.isnan))
sel_AGNLINER_NIIHa_OIIIHb = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01)
sel_SF_NIIHa_OIIIHb_K01 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_K01)
sel_SF_NIIHa_OIIIHb_K03 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_K03)
sel_SF_NIIHa_OIIIHb_S06 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_S06)
sel_AGN_SIIHa_OIIIHb_K01 = sel_SIIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_SII)
sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01
sel_AGN_OIHa_OIIIHb_K01 = sel_OIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_OI)
sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01
sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01 & sel_AGN_OIHa_OIIIHb_K01
sel_AGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > EW_hDIG*bug))
sel_SAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > EW_strong*bug))
sel_VSAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > EW_verystrong*bug))
sel_pAGB = sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen <= EW_hDIG)
sel_SF_EW = sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen > EW_SF*bug)
###############################################################################
# END SETTING VALUES ##########################################################
###############################################################################

###############################################################################
# BEGIN REPORTS ###############################################################
###############################################################################
###############################################################################
# BEGIN REPORT RATIOS #########################################################
###############################################################################
print '\n#RR#################'
print '#RR# REPORT RATIOS #'
print '#RR#################'
groups = [
    ['log_NII_Ha_cen_mean'],
    ['log_SII_Ha_cen_mean'],
    ['log_OI_Ha_cen'],
    ['log_OIII_Hb_cen_mean'],
    ['log_NII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
    ['log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
    ['log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
    ['log_NII_Ha_cen_mean', 'log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
    ['log_NII_Ha_cen_mean', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
    ['log_NII_Ha_cen_mean', 'log_SII_Ha_cen_mean', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
]
for g in groups:
    if len(g) > 1:
        N = elines.groupby(g).ngroups
    else:
        N = elines[g[0]].count()
    print '#RR# %s measured: %d galaxies' % (g, N)
print '#RR#################\n'
###############################################################################
# END REPORT RATIOS ###########################################################
###############################################################################

###############################################################################
# BEGIN REPORT AGN CANDIDATES #################################################
###############################################################################
print '\n#AC##################'
print '#AC# AGN CANDIDATES #'
N_TOT = len(elines.index)
N_GAS = len(elines.loc[~(elines['log_NII_Ha_cen_mean'].apply(np.isnan)) & ~(elines['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
N_NO_GAS = N_TOT - N_GAS
###############################################################
# [OIII] vs [NII]
###############################################################
# AGN/LINER
###############################################################
m = sel_AGNLINER_NIIHa_OIIIHb
N_AGN_NII_Ha = m.values.astype('int').sum()
# plus EW(Ha)
N_AGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
N_SAGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
N_VSAGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
# SF
m = sel_SF_NIIHa_OIIIHb_K01
N_SF_K01 = m.values.astype('int').sum()
N_SF_K01_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
N_SSF_K01 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
N_VSSF_K01 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
m = sel_SF_NIIHa_OIIIHb_K03
N_SF_K03 = m.values.astype('int').sum()
N_SF_K03_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
elines.loc[(m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)), 'TYPE'] = 6
N_SSF_K03 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
elines.loc[(m & sel_EW & (EW_Ha_cen > EW_strong*bug)), 'TYPE'] = 1
N_VSSF_K03 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
m = sel_SF_NIIHa_OIIIHb_S06
N_SF_S06 = m.values.astype('int').sum()
N_SF_S06_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
N_SSF_S06 = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
N_VSSF_S06 = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
###############################################################
# [OIII] vs [NII] + [OIII] vs [SII]
###############################################################
m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01
N_AGN_NII_SII_Ha = m.values.astype('int').sum()
# plus EW(Ha)
N_AGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
N_SAGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
N_VSAGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
###############################################################
# [OIII] vs [NII] + [OIII] vs [OI]
###############################################################
m = sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01
N_AGN_NII_OI_Ha = m.values.astype('int').sum()
# plus EW(Ha)
N_AGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
N_SAGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
N_VSAGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
###############################################################
# [OIII] vs [NII] + [OIII] vs [SII] + [OIII] vs [OI]
###############################################################
###############################################################
m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01
N_AGN_NII_SII_OI_Ha = m.values.astype('int').sum()
# plus EW(Ha)
N_AGN = (m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)).values.astype('int').sum()
elines.loc[(m & sel_EW & (EW_Ha_cen > EW_hDIG*bug)), 'TYPE'] = 2
N_SAGN = (m & sel_EW & (EW_Ha_cen > EW_strong*bug)).values.astype('int').sum()
elines.loc[(m & sel_EW & (EW_Ha_cen > EW_strong*bug)), 'TYPE'] = 3
N_VSAGN = (m & sel_EW & (EW_Ha_cen > EW_verystrong*bug)).values.astype('int').sum()
###############################################################
###############################################################
# pAGB
###############################################################
m = sel_pAGB
N_pAGB = m.values.astype('int').sum()
elines.loc[m, 'TYPE'] = 4
N_pAGB_aboveK01 = (m & (log_OIII_Hb_cen > y_mod_K01)).values.astype('int').sum()
N_pAGB_aboveK03 = (m & (log_OIII_Hb_cen > y_mod_K03)).values.astype('int').sum()
N_pAGB_aboveS06 = (m & (log_OIII_Hb_cen > y_mod_S06)).values.astype('int').sum()
###############################################################
###############################################################
# SF
###############################################################
m = sel_SF_EW
N_SF_EW = (m).values.astype('int').sum()
###############################################################
m = (elines['TYPE'] == 2) | (elines['TYPE'] == 3)
elines.loc[m, 'AGN_FLAG'] = 2
m = ((elines['SN_broad'] > 8) & ((elines['TYPE'] == 2) | (elines['TYPE'] == 3))) | (elines['broad_by_eye'] == 1)
m = ((elines['SN_broad'] > 8) & (elines['AGN_FLAG'] == 2)) | (elines['broad_by_eye'] == 1)
elines.loc[m, 'AGN_FLAG'] = 1
columns_to_csv = [
    'AGN_FLAG', 'SN_broad',
    'RA', 'DEC', 'log_NII_Ha_cen_mean', 'log_NII_Ha_cen_stddev',
    'log_OIII_Hb_cen_mean', 'log_OIII_Hb_cen_stddev',
    'log_SII_Ha_cen_mean', 'log_SII_Ha_cen_stddev',
    'log_OI_Ha_cen', 'e_log_OI_Ha_cen',
    'EW_Ha_cen_mean', 'EW_Ha_cen_stddev',
]
elines.loc[elines['AGN_FLAG'] > 0].to_csv('AGN_CANDIDATES.csv', columns=columns_to_csv)
# OUTPUT ######################################################################
print '#AC##################'
print '#AC# %sN.TOTAL%s = %d' % (color.B, color.E, N_TOT)
print '#AC# %sN.NO GAS%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS)
print '#AC##################'
print '#AC# %sEW cuts%s:' % (color.B, color.E)
print '#AC# \t%snot-pAGB%s (%sN%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, EW_hDIG, bug)
print '#AC# \t%sStrong%s (%sS%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, EW_strong, bug)
print '#AC# \t%sVery strong%s (%sVS%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, EW_verystrong, bug)
print '#AC# \t%sSF%s: EW > %d A' % (color.B, color.E, EW_SF)
print '#AC##################'
print '#AC# N.AGNs/LINERs candidates by [NII]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_Ha, color.B, color.E, N_AGN_NII_Ha_EW, color.B, color.E, N_SAGN_NII_Ha_EW, color.B, color.E, N_VSAGN_NII_Ha_EW)
print '#AC# N.AGNs candidates by [NII]/Ha and [SII]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_Ha, color.B, color.E, N_AGN_NII_SII_Ha_EW, color.B, color.E, N_SAGN_NII_SII_Ha_EW, color.B, color.E, N_VSAGN_NII_SII_Ha_EW)
print '#AC# N.AGNs candidates by [NII]/Ha and [OI]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_OI_Ha, color.B, color.E, N_AGN_NII_OI_Ha_EW, color.B, color.E, N_SAGN_NII_OI_Ha_EW, color.B, color.E, N_VSAGN_NII_OI_Ha_EW)
print '#AC# N.AGNs candidates by [NII]/Ha, [SII]/Ha and [OI]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_OI_Ha, color.B, color.E, N_AGN, color.B, color.E, N_SAGN, color.B, color.E, N_VSAGN)
print '#AC# N.AGNs %sType-II%s: %d - %sType-I%s: %d' % (color.B, color.E, elines['AGN_FLAG'].loc[elines['AGN_FLAG']==2].count(), color.B, color.E, elines['AGN_FLAG'].loc[elines['AGN_FLAG']==1].count())
print '#AC# N.pAGB: %d (%sabove K01%s: %d - %sabove K03%s: %d - %sabove S06%s: %d)' % (N_pAGB, color.B, color.E, N_pAGB_aboveK01, color.B, color.E, N_pAGB_aboveK03, color.B, color.E, N_pAGB_aboveK03)
print '#AC# N_SF %sEW%s: %d' % (color.B, color.E, N_SF_EW)
print '#AC# N.SF %sK01%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K01, color.B, color.E, N_SF_K01_EW, color.B, color.E, N_SSF_K01, color.B, color.E, N_VSSF_K01)
print '#AC# N.SF %sK03%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K03, color.B, color.E, N_SF_K03_EW, color.B, color.E, N_SSF_K03, color.B, color.E, N_VSSF_K03)
print '#AC# N.SF %sS06%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_S06, color.B, color.E, N_SF_S06_EW, color.B, color.E, N_SSF_S06, color.B, color.E, N_VSSF_S06)
print '#AC##################\n'
###############################################################################
# END REPORTS #################################################################
###############################################################################

###############################################################################
# BEGIN PLOTS #################################################################
###############################################################################
if plot:
    latex_ppi = 72.0
    latex_column_width_pt = 240.0
    latex_column_width = latex_column_width_pt/latex_ppi
    latex_text_width_pt = 504.0
    latex_text_width = latex_text_width_pt/latex_ppi
    golden_mean = 0.5 * (1. + 5**0.5)
    ##########################
    mtI = elines['AGN_FLAG'] == 1
    mtII = elines['AGN_FLAG'] == 2
    color_AGN_tI = 'k'
    color_AGN_tII = 'mediumslateblue'
    scatter_kwargs = dict(s=2, vmax=2.5, vmin=-0.5, cmap='viridis_r', marker='o', edgecolor='none')
    scatter_AGN_tII_kwargs = dict(s=50, linewidth=0.1, marker='*', facecolor='none', edgecolor=color_AGN_tII)
    scatter_AGN_tI_kwargs = dict(s=50, linewidth=0.1, marker='*', facecolor='none', edgecolor=color_AGN_tI)
    legend_elements = [
        Line2D([0], [0], marker='*', markeredgecolor=color_AGN_tI, label='Type-I AGN', markerfacecolor='none', markersize=7, markeredgewidth=0.12, linewidth=0),
        Line2D([0], [0], marker='*', markeredgecolor=color_AGN_tII, label='Type-II AGN', markerfacecolor='none', markersize=7, markeredgewidth=0.12, linewidth=0),
    ]
    ##########################

    ############################
    # PLOT ANCILLARY FUNCTIONS #
    ############################
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


    def plot_colored_by_EW(x, y, z, xlabel=None, ylabel=None, extent=None, n_bins_maj_x=5, n_bins_maj_y=5, n_bins_min_x=5, n_bins_min_y=5, prune_x='upper', prune_y=None, verbose=False, output_name=None, markAGNs=False, f=None, ax=None):
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
            ax.set_xlabel(xlabel, fontsize=fs+1)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fs+1)
        cb_width = 0.05
        cb_ax = f.add_axes([right, bottom, cb_width, top-bottom])
        cb = plt.colorbar(sc, cax=cb_ax)
        cb.set_label(r'$\log\ |{\rm W}_{{\rm H}\alpha}|$', fontsize=fs+1)
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
        ax.grid(linestyle='--', color='gray', linewidth=0.1, alpha=0.3)
        if verbose:
            print '# x #'
            xlim = ax.get_xlim()
            x_low = x.loc[x < xlim[0]]
            x_upp = x.loc[x > xlim[1]]
            print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '#####'
            print '# y #'
            ylim = ax.get_ylim()
            y_low = y.loc[y < ylim[0]]
            y_upp = y.loc[y > ylim[1]]
            print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '#####'
        if output_name is not None:
            f.savefig(output_name, dpi=_dpi_choice, transparent=_transp_choice)
        return f, ax
    ############################


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
    fs = 7
    y = log_OIII_Hb_cen
    ##########################
    ### NII/Ha
    print '##########################'
    print '## [NII]/Ha             ##'
    print '##########################'
    ax = ax0
    x = log_NII_Ha_cen
    extent = [-1.6, 0.8, -1.2, 1.5]
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs)
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
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs)
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
    sc = ax.scatter(x, y, c=EW_Ha_cen.apply(np.log10), **scatter_kwargs)
    ax.scatter(x.loc[mtII], y[mtII], **scatter_AGN_tII_kwargs)
    ax.scatter(x.loc[mtI], y[mtI], **scatter_AGN_tI_kwargs)
    ax.set_xlabel(r'$\log\ ({\rm [OI]}/{\rm H\alpha})$', fontsize=fs+4)
    cb_ax = f.add_axes([right, bottom, 0.02, top-bottom])
    cb = plt.colorbar(sc, cax=cb_ax)
    cb.set_label(r'$\log\ |{\rm W}_{{\rm H}\alpha}|$', fontsize=fs+4)
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
    ##########################
    f.text(0.01, 0.5, r'$\log\ ({\rm [OIII]}/{\rm H\beta})$', va='center', rotation='vertical', fontsize=fs+4)
    f.savefig('fig_BPT.%s' % img_suffix, dpi=_dpi_choice, transparent=_transp_choice)
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
    extent = [8, 13, -4.5, 2.5]
    n_bins_min_x = 2
    n_bins_maj_y = 4
    n_bins_min_y = 2
    prune_x = None
    plot_colored_by_EW(x=x, y=elines['lSFR'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                       ylabel=r'$\log ({\rm SFR}/{\rm M}_{\odot}/{\rm yr})$',
                       xlabel=xlabel, extent=extent,
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=verbose,
                       output_name='fig_SFMS.%s' % img_suffix)
    print '###########################\n'
    print '\n####################################'
    print '## SFMS colored by EW_Ha (NO CEN) ##'
    print '####################################'
    plot_colored_by_EW(x=x, y=elines['lSFR_NO_CEN'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                       ylabel=r'$\log ({\rm SFR}/{\rm M}_{\odot}/{\rm yr})_{NO CEN}$',
                       xlabel=xlabel, extent=extent,
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=verbose,
                       output_name='fig_SFMS_NC.%s' % img_suffix)
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
    plot_colored_by_EW(x=elines['log_Mass'], y=elines['C'], z=EW_Ha_cen.apply(np.log10), markAGNs=True,
                       xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                       ylabel=r'$\log ({\rm R}90/{\rm R}50)$',
                       extent=[8, 13, 0.5, 5.5],
                       n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                       n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                       verbose=verbose,
                       output_name='fig_M_C.%s' % img_suffix)
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
    output_name = 'fig_sSFR_C.%s' % img_suffix
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_EW(f=f, ax=ax, x=elines['lSFR'] - elines['log_Mass'], y=elines['C'], z=EW_Ha_cen.apply(np.log10),
                               xlabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                               ylabel=r'$\log ({\rm R}90/{\rm R}50)$',
                               extent=[-13.5, -8.5, 0.5, 5.5], markAGNs=True,
                               n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                               n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                               verbose=verbose)
    ax.axvline(x=-11.8, c='k', ls='--')
    ax.axvline(x=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=_dpi_choice, transparent=_transp_choice)
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
    output_name = 'fig_M_sSFR.%s' % img_suffix
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    N_rows, N_cols = 1, 1
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    f, ax = plot_colored_by_EW(f=f, ax=ax, y=elines['lSFR'] - elines['log_Mass'], x=elines['log_Mass'], z=EW_Ha_cen.apply(np.log10),
                               ylabel=r'$\log ({\rm sSFR}_\star/{\rm yr})$',
                               xlabel=r'$\log ({\rm M}_\star/{\rm M}_{\odot})$',
                               extent=[8, 13, -13.5, -8.5], markAGNs=True,
                               n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y,
                               n_bins_min_x=n_bins_min_x, prune_x=prune_x,
                               verbose=verbose)
    ax.axhline(y=-11.8, c='k', ls='--')
    ax.axhline(y=-10.8, c='k', ls='--')
    f.savefig(output_name, dpi=_dpi_choice, transparent=_transp_choice)
    print '#############################\n'
    ##########################

    ################################
    ## X Y histo colored by EW_Ha ##
    ################################
    def plot_histo_xy_colored_by_EW(x, y, z, ax_Hx, ax_Hy, ax_sc, xlabel=None, xrange=None, n_bins_maj_x=5, n_bins_min_x=5, prune_x=None, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, verbose=False):
        ax_Hx.hist(x, bins=20, range=xrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
        ax_Hx.hist(x[mtI], bins=20, range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
        ax_Hx.hist(x[mtII], bins=20, hatch='//////', range=xrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
        ax_Hx.set_xlabel(xlabel)
        ax_Hx.set_xlim(xrange)
        ax_Hx.xaxis.set_major_locator(MaxNLocator(n_bins_maj_x, prune=prune_x))
        ax_Hx.xaxis.set_minor_locator(AutoMinorLocator(n_bins_min_x))
        ax_Hx.set_ylim(0, 1)
        ax_Hx.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
        ax_Hx.yaxis.set_minor_locator(AutoMinorLocator(5))
        tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=True, labelbottom=True, labeltop=False, labelleft=False, labelright=True)
        ax_Hx.tick_params(**tick_params)
        ####################################
        ax_Hy.hist(y, orientation='horizontal', bins=20, range=yrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
        ax_Hy.hist(y[mtI], orientation='horizontal', bins=20, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
        ax_Hy.hist(y[mtII], orientation='horizontal', bins=20, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
        # ax_Hy.hist(y[mtI], orientation='horizontal', hatch='////', bins=20, range=yrange, histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
        # ax_Hy.hist(y[mtII], orientation='horizontal', hatch='//', bins=20, range=yrange, histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
        ax_Hy.set_ylabel(ylabel)
        ax_Hy.set_ylim(yrange)
        ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
        ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
        ax_Hy.set_xlim(0, 1)
        ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
        ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
        tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
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
        cb.set_label(r'$\log\ |{\rm W}_{{\rm H}\alpha}|$', fontsize=fs+1)
        cb.locator = MaxNLocator(4)
        # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
        cb_ax.tick_params(which='both', direction='in')
        cb.update_ticks()
        ####################################
        if verbose:
            print '# x #'
            xlim = ax_sc.get_xlim()
            x_low = x.loc[x < xlim[0]]
            x_upp = x.loc[x > xlim[1]]
            print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '#####'
            print '# y #'
            ylim = ax_sc.get_ylim()
            y_low = y.loc[y < ylim[0]]
            y_upp = y.loc[y > ylim[1]]
            print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines.loc[i, 'AGN_FLAG'])
            print '#####'
        return ax_Hx, ax_Hy, ax_sc

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
        'fig_histo_CMD_NSA': [elines['Mabs_R'], r'${\rm M}_{\rm R}$ (mag)', 5, 2, None, elines['B_R'], r'${\rm B-R}$ (mag)', 3, 5, None, EW_Ha_cen.apply(np.log10), [-24, -10, 0, 1.5]],
        ################################

        ##################################
        ## CMD (CUBES) colored by EW_Ha ##
        ##################################
        'fig_histo_CMD_CUBES': [elines.loc[m_redshift, 'Mabs_i'], r'${\rm M}_{\rm i}$ (mag)', 5, 2, None, elines.loc[m_redshift, 'u'] - elines.loc[m_redshift, 'i'], r'${\rm u}-{\rm i}$ (mag)', 3, 5, None, EW_color_zcut, [-24, -10, 0, 3.5]],
        ##################################

        ###########################
        ## SFMS colored by EW_Ha ##
        ###########################
        'fig_histo_SFMS': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})$', 4, 2, None, EW_Ha_cen.apply(np.log10), [8, 13, -4.5, 2.5]],
        ###########################

        ####################################
        ## SFMS colored by EW_Ha (NO CEN) ##
        ####################################
        'fig_histo_SFMS_NC': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR_NO_CEN'], r'$\log ({\rm SFR}_\star/{\rm M}_{\odot}/{\rm yr})_{NO CEN}$', 4, 2, None, EW_Ha_cen.apply(np.log10), [8, 13, -4.5, 2.5]],
        ####################################

        ##########################
        ## M-C colored by EW_Ha ##
        ##########################
        'fig_histo_M_C': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['C'], r'$\log ({\rm R}90/{\rm R}50)$', 6, 2, None, EW_Ha_cen.apply(np.log10), [8, 13, 0.5, 5.5]],
        ##########################

        #############################
        ## sSFR-C colored by EW_Ha ##
        #############################
        'fig_histo_sSFR_C': [elines['lSFR'] - elines['log_Mass'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None, elines['C'], r'$\log ({\rm R}90/{\rm R}50)$', 6, 2, None, EW_Ha_cen.apply(np.log10), [-13.5, -8.5, 0.5, 5.5]],
        #############################

        #############################
        ## M-sSFR colored by EW_Ha ##
        #############################
        'fig_histo_M_sSFR': [elines['log_Mass'], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 5, 2, None, elines['lSFR'] - elines['log_Mass'], r'$\log ({\rm sSFR}_\star/{\rm yr})$', 5, 2, None, EW_Ha_cen.apply(np.log10), [8, 13, -13.5, -8.5]],
        #############################
    }

    for k, v in plot_histo_xy_dict.iteritems():
        print '\n################################'
        print '# %s' % k
        x, xlabel, n_bins_maj_x, n_bins_min_x, prune_x = v[0:5]
        y, ylabel, n_bins_maj_y, n_bins_min_y, prune_y = v[5:10]
        extent = v[-1]
        z = v[-2]
        output_name = '%s.%s' % (k, img_suffix)
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        plot_histo_xy_colored_by_EW(x, y, z, ax_Hx, ax_Hy, ax_sc, xlabel=xlabel, xrange=extent[0:2], n_bins_maj_x=n_bins_maj_x, n_bins_min_x=n_bins_min_x, prune_x=prune_x, ylabel=ylabel, yrange=extent[2:4], n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=prune_y, verbose=verbose)
        if k == 'fig_histo_sSFR_C':
            ax_sc.axvline(x=-11.8, c='k', ls='--')
            ax_sc.axvline(x=-10.8, c='k', ls='--')
        if k == 'fig_histo_M_sSFR':
            ax_sc.axhline(y=-11.8, c='k', ls='--')
            ax_sc.axhline(y=-10.8, c='k', ls='--')
        f.savefig(output_name, dpi=_dpi_choice, transparent=_transp_choice)
        print '################################\n'
    print '################################\n'
    ################################
    ################################


    ###########
    ## Morph ##
    ###########
    # Create an object spanning only galaxies with defined morphology
    elines_wmorph = elines.loc[elines['morph'] >= 0].copy()
    N_TOT_WMORPH = len(elines_wmorph.index)
    N_GAS_WMORPH = len(elines_wmorph.loc[~(elines_wmorph['log_NII_Ha_cen_mean'].apply(np.isnan)) & ~(elines_wmorph['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
    N_NO_GAS_WMORPH = N_TOT - N_GAS
    print '###################'
    print '## Morph studies ##'
    print '###################\n'
    print '#AC##################'
    print '#AC# %sN.TOTAL WITH MORPHOLOGY%s = %d' % (color.B, color.E, N_TOT_WMORPH)
    print '#AC# %sN.NO GAS WITH MORPHOLOGY%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS_WMORPH)
    print '#AC##################\n'
    EW_color = elines_wmorph['EW_Ha_cen_mean'].apply(np.abs).apply(np.log10)
    mtI = elines_wmorph['AGN_FLAG'] == 1
    mtII = elines_wmorph['AGN_FLAG'] == 2
    scatter_kwargs = dict(c=EW_color, s=2, vmax=2.5, vmin=-0.5, cmap='viridis_r', marker='o', edgecolor='none')
    # Created to adjust the morphtype of ellipticals
    def morph_adjust(x):
        r = x
        # If not NAN or M_TYPE E* (0-7) call it E
        if ~np.isnan(x) and x <= 7:
            r = 7
        return r

    morph = elines_wmorph['morph'].apply(morph_adjust)

    def plot_x_morph(ax, verbose):
        x = morph
        H = ax_Hx.hist(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5], histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
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
        ax.set_ylim(0, 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(2, prune='upper'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=False, left=True, right=True, labelbottom=True, labeltop=False, labelleft=False, labelright=True)
        ax.tick_params(**tick_params)
        ####################################
        if verbose:
            print '# x #'
            xlim = ax.get_xlim()
            x_low = x.loc[x < xlim[0]]
            x_upp = x.loc[x > xlim[1]]
            print '# N.x points < %.1f: %d' % (xlim[0], x_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_low.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
            print '# N.x points > %.1f: %d' % (xlim[1], x_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in x_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, x_upp.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
            print '#####'
        return ax


    def plot_morph_y_colored_by_EW(y, ax_Hx, ax_Hy, ax_sc, ylabel=None, yrange=None, n_bins_maj_y=5, n_bins_min_y=5, prune_y=None, verbose=False):
        # ax_Hy.hist(y, orientation='horizontal', bins=20, range=yrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True)
        ax_Hy.hist(y, orientation='horizontal', bins=20, range=yrange, histtype='step', fill=True, facecolor='green', edgecolor='none', align='mid', density=True, alpha=0.5)
        m = np.linspace(7, 19, 13).astype('int')
        y_mean = np.array([y.loc[morph == mt].mean() for mt in m])
        # ax_Hy.hist(y[mtI], orientation='horizontal', hatch='////', bins=20, range=yrange, histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
        ax_Hy.hist(y[mtI], orientation='horizontal', bins=20, range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tI, align='mid', density=True)
        N_y_tI_above = np.array([np.array(y.loc[mtI & (morph == mt)] > y.loc[mtI & (morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
        print '# Type-I AGN above mean: %d (%.1f%%)' % (N_y_tI_above, 100.*N_y_tI_above/y[mtI].count())
        # ax_Hy.hist(y[mtII], orientation='horizontal', hatch='//', bins=20, range=yrange, histtype='step', fill=False, facecolor='none', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
        ax_Hy.hist(y[mtII], orientation='horizontal', bins=20, hatch='//////', range=yrange, histtype='step', linewidth=1, edgecolor=color_AGN_tII, align='mid', density=True)
        N_y_tII_above = np.array([np.array(y.loc[mtII & (morph == mt)] > y.loc[mtII & (morph == mt)].mean()).astype('int').sum() for mt in m]).sum()
        print '# Type-II AGN above mean: %d (%.1f%%)' % (N_y_tII_above, 100.*N_y_tII_above/y[mtII].count())
        ax_Hy.set_ylabel(ylabel)
        # ax_Hy.set_xlim(0, 1)
        ax_Hy.xaxis.set_major_locator(MaxNLocator(2, prune='upper'))
        ax_Hy.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_Hy.set_ylim(yrange)
        ax_Hy.yaxis.set_major_locator(MaxNLocator(n_bins_maj_y, prune=prune_y))
        ax_Hy.yaxis.set_minor_locator(AutoMinorLocator(n_bins_min_y))
        tick_params = dict(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
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
        cb.set_label(r'$\log\ |{\rm W}_{{\rm H}\alpha}|$', fontsize=fs+1)
        cb.locator = MaxNLocator(4)
        # cb_ax.tick_params(which='both', direction='out', pad=13, left=True, right=False)
        cb_ax.tick_params(which='both', direction='in')
        cb.update_ticks()
        ####################################
        if verbose:
            print '# y #'
            ylim = ax_sc.get_ylim()
            y_low = y.loc[y < ylim[0]]
            y_upp = y.loc[y > ylim[1]]
            print '# N.y points < %.1f: %d' % (ylim[0], y_low.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_low.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_low.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
            print '# N.y points > %.1f: %d' % (ylim[1], y_upp.count())
            if type(verbose) is str and verbose > 'v':
                for i in y_upp.index:
                    print '#\t%s: %.3f (AGN:%d)' % (i, y_upp.loc[i], elines_wmorph.loc[i, 'AGN_FLAG'])
            print '#####'
        return ax_Hx, ax_Hy, ax_sc


    ############################
    ## Morph colored by EW_Ha ##
    ############################
    print '\n############################'
    print '## Morph colored by EW_Ha ##'
    print '############################'
    plots_dict = {
        'fig_Morph_M': ['log_Mass', [8, 13], r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', 6, 2],
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
        output_name = '%s.%s' % (k, img_suffix)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax_Hx = plt.subplot(gs[-1, 1:])
        ax_Hy = plt.subplot(gs[0:3, 0])
        ax_sc = plt.subplot(gs[0:-1, 1:])
        ax_Hx = plot_x_morph(ax_Hx, verbose)
        plot_morph_y_colored_by_EW(y, ax_Hx, ax_Hy, ax_sc, ylabel=ylabel, yrange=yrange, n_bins_maj_y=n_bins_maj_y, n_bins_min_y=n_bins_min_y, prune_y=None, verbose=verbose)
        # gs.tight_layout(f)
        f.savefig(output_name, dpi=_dpi_choice, transparent=_transp_choice)
        print '############################\n'
    print '############################\n'
    ############################

###############################################################################
# END PLOTS ###################################################################
###############################################################################
