#!/usr/bin/python3
import os
import sys
import pickle
import itertools
import numpy as np
import pandas as pd
from pytu.functions import debug_var
from pytu.objects import readFileArgumentParser
from CALIFAUtils.scripts import spaxel_size_pc, redshift_dist_Mpc

morph_name = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'I']


def parser_args(default_args_file='args/default_tables.args'):
    """
    Parse the command line args.

    With fromfile_pidxrefix_chars=@ we can read and parse command line args
    inside a file with @file.txt.
    default args inside default_args_file
    """
    default_args = {
        'output': 'dataframes.pkl',
        'csv_dir': 'csv',
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--output', '-O', metavar='FILE', type=str, default=default_args['output'])
    parser.add_argument('--csv_dir', metavar='DIR', type=str, default=default_args['csv_dir'])
    parser.add_argument('--verbose', '-v', action='count')
    args_list = sys.argv[1:]
    # if exists file default.args, load default args
    if os.path.isfile(default_args_file):
        args_list.insert(0, '@%s' % default_args_file)
    debug_var(True, args_list=args_list)
    args = parser.parse_args(args=args_list)
    args = parser.parse_args(args=args_list)
    debug_var(True, args=args)
    return args


args = parser_args()

# Files and directories
fname1 = 'CALIFA_3_DR4_classnum.pandas.csv'
fname2 = 'CALIFA_basic_joint.pandas.csv'
fname3 = 'get_CALIFA_cen_broad.pandas.csv'
fname4 = 'get_mag_cubes_v2.2.pandas.csv'
fname5 = 'get_mag_cubes_v2.2.NO_CEN.pandas.csv'
fname6 = 'get_RA_DEC.pandas.csv'
fname7 = 'get_proc_elines_CALIFA.all_good.pandas.csv'
fname8 = 'NII_Ha_fit.csv'
fname9 = 'bitsakis_t12.csv'
fname10 = 'get_new_morph_temp.pandas.csv'
fname11 = 'AGNs_CALIFA_weights_gals.csv'
fnames_short = {
    fname1: 'DR4_morph',
    fname2: 'basic_joint',
    fname3: 'cen_broad',
    fname4: 'mag_cubes_v2.2',
    fname5: 'mag_cubes_v2.2.NC',
    fname6: 'RA_DEC',
    fname7: 'elines',
    fname8: 'broad_fit',
    fname9: 'bitsakis_t12',
    fname10: 'new_morph',
    fname11: 'weights'
}
fnames_long = {
    'DR4_morph': fname1,
    'basic_joint': fname2,
    'cen_broad': fname3,
    'mag_cubes_v2.2': fname4,
    'mag_cubes_v2.2.NC': fname5,
    'RA_DEC': fname6,
    'elines': fname7,
    'broad_fit': fname8,
    'bitsakis_t12': fname9,
    'new_morph': fname10,
    'weights': fname11,
}
# Read CSV files
df = {}
na_values = ['BAD', 'nan', -999, '-inf', 'inf']
for k, v in fnames_short.items():
    f_path = '%s/%s' % (args.csv_dir, k)
    # print(f_path)
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
df['elines']['broad_by_eye'] = False
with open('%s/list_Broad_by_eye.pandas.csv' % args.csv_dir, 'r') as f:
    for l in f.readlines():
        if l[0] != '#':
            DBName = l.strip()
            if DBName in df['elines'].index:
                df['elines'].loc[DBName, 'broad_by_eye'] = True
                print('%s: broad-line by eye' % (DBName))
            else:
                print('%s: not in %s' % (DBName, fnames_long['elines']))

# Populating dataframe elines joining different data from other dataframes
for c in df['bitsakis_t12'].columns[1:]:
    df['elines'][c] = df['bitsakis_t12'][c]
df['elines']['CALIFAID'] = df['elines']['CALIFAID']
df['elines']['CALIFAID_2'] = df['basic_joint']['CALIFAID']
df['elines'].rename(columns={'lSFR_NO_CEN': 'lSFR_NC'}, inplace=True)
df['elines']['C'] = df['mag_cubes_v2.2']['C']
df['elines']['e_C'] = df['mag_cubes_v2.2']['error_C']
df['elines']['Mabs_V'] = df['mag_cubes_v2.2']['V_band_abs_mag']
df['elines']['e_Mabs_V'] = df['mag_cubes_v2.2']['V_band_abs_mag_error']
df['elines']['Mabs_R'] = df['mag_cubes_v2.2']['R_band_abs_mag']
df['elines']['e_Mabs_R'] = df['mag_cubes_v2.2']['R_band_abs_mag_error']
df['elines']['Mabs_i'] = df['mag_cubes_v2.2']['i_band_abs_mag']
df['elines']['e_Mabs_i'] = df['mag_cubes_v2.2']['i_band_abs_mag_error']
df['elines']['Mabs_r'] = df['mag_cubes_v2.2']['r_band_abs_mag']
df['elines']['e_Mabs_r'] = df['mag_cubes_v2.2']['r_band_abs_mag_error']
df['elines']['B_V'] = df['mag_cubes_v2.2']['B_V']
df['elines']['e_B_V'] = df['mag_cubes_v2.2']['error_B_V']
df['elines']['B_R'] = df['mag_cubes_v2.2']['B_R']
df['elines']['e_B_R'] = df['mag_cubes_v2.2']['error_B_R']
df['elines']['u'] = df['mag_cubes_v2.2']['u_band_mag']
df['elines']['g'] = df['mag_cubes_v2.2']['g_band_mag']
df['elines']['r'] = df['mag_cubes_v2.2']['r_band_mag']
df['elines']['i'] = df['mag_cubes_v2.2']['i_band_mag']
df['elines']['u_abs'] = df['mag_cubes_v2.2']['u_band_abs_mag']
df['elines']['g_abs'] = df['mag_cubes_v2.2']['g_band_abs_mag']
df['elines']['r_abs'] = df['mag_cubes_v2.2']['r_band_abs_mag']
df['elines']['i_abs'] = df['mag_cubes_v2.2']['i_band_abs_mag']
df['elines']['Mabs_V_NC'] = df['mag_cubes_v2.2.NC']['V_band_abs_mag']
df['elines']['e_Mabs_V_NC'] = df['mag_cubes_v2.2.NC']['V_band_abs_mag_error']
df['elines']['Mabs_R_NC'] = df['mag_cubes_v2.2.NC']['R_band_abs_mag']
df['elines']['e_Mabs_R_NC'] = df['mag_cubes_v2.2.NC']['R_band_abs_mag_error']
df['elines']['Mabs_i_NC'] = df['mag_cubes_v2.2.NC']['i_band_abs_mag']
df['elines']['e_Mabs_i_NC'] = df['mag_cubes_v2.2.NC']['i_band_abs_mag_error']
df['elines']['Mabs_r_NC'] = df['mag_cubes_v2.2.NC']['r_band_abs_mag']
df['elines']['e_Mabs_r_NC'] = df['mag_cubes_v2.2.NC']['r_band_abs_mag_error']
df['elines']['B_V_NC'] = df['mag_cubes_v2.2.NC']['B_V']
df['elines']['e_B_V_NC'] = df['mag_cubes_v2.2.NC']['error_B_V']
df['elines']['B_R_NC'] = df['mag_cubes_v2.2.NC']['B_R']
df['elines']['e_B_R_NC'] = df['mag_cubes_v2.2.NC']['error_B_R']
df['elines']['u_NC'] = df['mag_cubes_v2.2.NC']['u_band_mag']
df['elines']['g_NC'] = df['mag_cubes_v2.2.NC']['g_band_mag']
df['elines']['r_NC'] = df['mag_cubes_v2.2.NC']['r_band_mag']
df['elines']['i_NC'] = df['mag_cubes_v2.2.NC']['i_band_mag']
df['elines']['u_abs_NC'] = df['mag_cubes_v2.2.NC']['u_band_abs_mag']
df['elines']['g_abs_NC'] = df['mag_cubes_v2.2.NC']['g_band_abs_mag']
df['elines']['r_abs_NC'] = df['mag_cubes_v2.2.NC']['r_band_abs_mag']
df['elines']['i_abs_NC'] = df['mag_cubes_v2.2.NC']['i_band_abs_mag']
df['elines']['redshift'] = df['mag_cubes_v2.2']['redshift']
df['elines']['redshift_CALIFA'] = df['basic_joint']['redshift_CALIFA']
df['elines']['morph'] = df['DR4_morph']['hubtyp']
df['elines']['RA'] = df['basic_joint']['ra']
df['elines']['DEC'] = df['basic_joint']['de']
df['elines']['RA'] = df['RA_DEC']['RA']
df['elines']['DEC'] = df['RA_DEC']['DEC']
df['elines']['bar'] = df['DR4_morph']['bar']
df['elines']['SN_broad'] = df['cen_broad']['Nsigma']
df['elines']['broad'] = 0
df['elines']['TYPE'] = 0
df['elines']['AGN_FLAG'] = 0
df['elines']['MORPH'] = 'none'
df['elines']['merg'] = df['DR4_morph']['merg']
df['elines']['GalaxyName'] = df['DR4_morph']['REALNAME']
df['elines']['GalaxyName'] = df['elines']['GalaxyName'].fillna('')
df['elines']['morph'] = df['elines']['morph'].fillna(-1)
df['elines']['Ha_broad'] = df['broad_fit']['Ha_broad']
df['elines']['Ha_narrow'] = df['broad_fit']['Ha_narrow']
df['elines']['NII_6583'] = df['broad_fit']['NII_6583']
df['elines']['NII_6548'] = df['broad_fit']['NII_6548']
df['elines']['EW_Ha_cen_mean'] = df['elines']['EW_Ha_cen_mean'].apply(np.abs)
df['elines']['log_EW_Ha_cen_mean'] = df['elines']['EW_Ha_cen_mean'].apply(np.log10)
df['elines']['EW_Ha_ALL'] = df['elines']['EW_Ha_ALL'].apply(np.abs)
df['elines']['log_EW_Ha_ALL'] = df['elines']['EW_Ha_ALL'].apply(np.log10)
df['elines']['EW_Ha_Re'] = df['elines']['EW_Ha_Re'].apply(np.abs)
df['elines']['log_EW_Ha_Re'] = df['elines']['EW_Ha_Re'].apply(np.log10)
df['elines'].loc[df['elines']['SN_broad'] <= 0, 'SN_broad'] = 0.
df['elines'].loc[df['elines']['log_Mass'] < 0, 'log_Mass'] = np.nan
df['elines'].loc[df['elines']['lSFR'] < -10, 'lSFR'] = np.nan
df['elines'].loc[df['elines']['lSFR_NC'] < -10, 'lSFR_NC'] = np.nan
df['elines'].loc[df['elines']['log_SFR_SF'] < -10, 'log_SFR_SF'] = np.nan
df['elines'].loc[df['elines']['log_SFR_ssp'] < -10, 'log_SFR_ssp'] = np.nan
df['elines'].loc[df['elines']['log_SFR_ssp_10Myr'] < -10, 'log_SFR_ssp_10Myr'] = np.nan
df['elines'].loc[df['elines']['log_SFR_ssp_100Myr'] < -10, 'log_SFR_ssp_100Myr'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas'] == -12, 'log_Mass_gas'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas_Re'] == -12, 'log_Mass_gas_Re'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas_Av_ssp_OH'] == -12, 'log_Mass_gas_Av_ssp_OH'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas_Av_ssp_ZH'] == -12, 'log_Mass_gas_Av_ssp_ZH'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas_Av_gas_rad'] == -12, 'log_Mass_gas_Av_gas_rad'] = np.nan
df['elines'].loc[df['elines']['log_Mass_gas_Av_gas_OH'] == -12, 'log_Mass_gas_Av_gas_OH'] = np.nan
df['elines']['log_NII_Ha_cen_fit'] = np.log10(df['elines']['NII_6583'] / df['elines']['Ha_narrow'])
# a, b = df['elines']['log_NII_Ha_cen_fit'], df['elines']['log_NII_Ha_cen_mean']
# m = a.apply(np.isnan)
# m = a.apply(np.isnan) | (df['elines']['EW_Ha_cen_mean'] <= 3)
# df['elines']['log_NII_Ha_cen'] = np.where(m, b, a)
# m = df['elines']['SN_broad'] > 5
# df['elines']['log_NII_Ha_cen'] = np.where(m, a, b)
df['elines']['log_NII_Ha_cen'] = df['elines']['log_NII_Ha_cen_mean']
# f = np.log10(df['elines']['F_Ha_cen']) - np.log10(df['elines']['Ha_narrow'])  # -16 +16
# df['elines']['log_SII_Ha_cen_fit'] = df['elines']['log_SII_Ha_cen_mean'] + f
# df['elines']['log_OI_Ha_cen_fit'] = df['elines']['log_OI_Ha_cen']  + f
df['elines']['Mass_cen'] = 10**df['elines']['Sigma_Mass_cen'] * spaxel_size_pc(redshift_dist_Mpc(df['elines']['z_stars'], 71), 3)**2.
df['elines']['log_Mass_corr_NC'] = np.log10(10**df['elines']['log_Mass_corr'] - df['elines']['Mass_cen'])
df['elines']['sSFR'] = df['elines']['lSFR'] - df['elines']['log_Mass_corr']
df['elines']['sSFR_NC'] = df['elines']['lSFR_NC'] - df['elines']['log_Mass_corr_NC']
df['elines']['sSFR_SF'] = df['elines']['log_SFR_SF'] - df['elines']['log_Mass_corr']
df['elines']['sSFR_ssp'] = df['elines']['log_SFR_ssp'] - df['elines']['log_Mass_corr']
df['elines']['sSFR_ssp_10Myr'] = df['elines']['log_SFR_ssp_10Myr'] - df['elines']['log_Mass_corr']
df['elines']['sSFR_ssp_100Myr'] = df['elines']['log_SFR_ssp_100Myr'] - df['elines']['log_Mass_corr']
df['elines']['e_log_Mass_gas'] = (1/np.log(10)) * df['elines']['e_Av_w'] / df['elines']['Av_w']
_x = df['elines']['Av_w']
_e_x = df['elines']['e_Av_w']
m = (_x > 0) & (_e_x > -1e300)
df['elines'].loc[~m, 'e_log_Mass_gas'] = np.nan

for gas_proxy in ['log_Mass_gas_Av_gas_rad','log_Mass_gas_Av_ssp_OH','log_Mass_gas_Av_gas_OH','log_Mass_gas','log_Mass_gas_Re','log_Mass_gas_Av_ssp_ZH']:
    df['elines']['SFE_%s' % gas_proxy] = df['elines']['lSFR'] - df['elines'][gas_proxy]
    df['elines']['SFE_SF_%s' % gas_proxy] = df['elines']['log_SFR_SF'] - df['elines'][gas_proxy]
    df['elines']['SFE_ssp_%s' % gas_proxy] = df['elines']['log_SFR_ssp'] - df['elines'][gas_proxy]
    df['elines']['log_tdep_%s' % gas_proxy] = df['elines'][gas_proxy] - df['elines']['lSFR'] - 9
    df['elines']['log_tdep_SF_%s' % gas_proxy] = df['elines'][gas_proxy] - df['elines']['log_SFR_SF'] - 9
    df['elines']['log_tdep_ssp_%s' % gas_proxy] = df['elines'][gas_proxy] - df['elines']['log_SFR_ssp'] - 9
    df['elines']['tdep_%s' % gas_proxy] = 10**df['elines']['log_tdep_%s' % gas_proxy]
    df['elines']['tdep_SF_%s' % gas_proxy] = 10**df['elines']['log_tdep_SF_%s' % gas_proxy]
    df['elines']['tdep_ssp_%s' % gas_proxy] = 10**df['elines']['log_tdep_ssp_%s' % gas_proxy]
    df['elines']['Mrat_%s' % gas_proxy] = 10**(df['elines']['log_Mass_corr'] - df['elines'][gas_proxy])
    df['elines']['fgas_%s' % gas_proxy] = 1 / (1 + df['elines']['Mrat_%s' % gas_proxy])
    df['elines']['log_fgas_%s' % gas_proxy] = df['elines']['fgas_%s' % gas_proxy].apply(np.log10)

df['elines']['delta_gas'] = df['elines']['log_Mass_gas_Av_gas_rad'] - df['elines']['log_Mass_gas']

# R. A. Calette volume correction
df['elines']['weights'] = df['weights']['w_califa']

filters = ['u', 'g', 'r', 'i']
for f1, f2 in itertools.combinations(filters, 2):
    df['elines']['%s_%s' % (f1, f2)] = df['elines']['%s' % f1] - df['elines']['%s' % f2]
    df['elines']['%s_%s_NC' % (f1, f2)] = df['elines']['%s_NC' % f1] - df['elines']['%s_NC' % f2]
df['elines']['NUV_r_SDSS'] = df['elines']['GALEX NUV [mag]'] - df['elines']['SDSS r [mag]']
df['elines']['NUV_r_CUBES'] = df['elines']['GALEX NUV [mag]'] - df['elines']['r']

for g in df['new_morph'].index:
    df['elines'].loc[g, 'morph'] = np.floor(df['new_morph'].loc[g, 'Average'])

# Create Morphology visual names array
for i in df['elines'].index:
    if df['elines'].loc[i, 'morph'] >= 0:
        df['elines'].loc[i, 'Morph'] = morph_name[df['elines'].loc[i, 'morph'].astype('int')]
df['elines']['Morph'] = df['elines']['Morph'].fillna('')

for c in df['elines'].columns:
    # mask inf and -inf elements in all numerical arrays
    if df['elines'][c].dtype.name[-2:] == '64':
        df['elines'][c] = df['elines'][c].mask(df['elines'][c].apply(np.isinf))

# Output pickled dataframe to args.output
with open(args.output, 'wb') as f:
    pickle.dump(df, f, protocol=2)

#####################
# check unique names
#####################
# gal_indexes = []
# for k in df.iterkeys():
#     x = set(df[k].index)
#     gal_indexes = list(set(gal_indexes) & x) + list(set(gal_indexes) ^ x)
# for n in sorted(gal_indexes):
#     print(n)
# sys.exit()

##########################
# MySQL dump for morph DB
##########################
# +------------+--------------+------+-----+---------+-------+
# | Field      | Type         | Null | Key | Default | Extra |
# +------------+--------------+------+-----+---------+-------+
# | GalID      | int(4)       | NO   | PRI | NULL    |       |
# | DBName     | varchar(50)  | NO   |     | NULL    |       |
# | GalaxyName | varchar(50)  | YES  |     | NULL    |       |
# | RA         | double       | YES  |     | NULL    |       |
# | DECL       | double       | YES  |     | NULL    |       |
# | BARRED     | tinyint(1)   | YES  |     | NULL    |       |
# | INTERACT   | tinyint(1)   | YES  |     | NULL    |       |
# | WEIGHT     | double       | YES  |     | NULL    |       |
# | comment    | varchar(255) | YES  |     | NULL    |       |
# | WarningID  | int(2)       | YES  | MUL | NULL    |       |
# | MorphID    | int(2)       | YES  | MUL | NULL    |       |
# +------------+--------------+------+-----+---------+-------+
###########################################################
# print(to create database to morphological classification)
###########################################################
# k=929
# print('INSERT INTO morph_class.to_classify (GalID, DBName, GalaxyName, RA, DECL, WEIGHT, comment, WarningID, MorphID) VALUES')
# for i, reg in df['elines'].iterrows():
#     m = int(reg['morph'])
#     if m < 0:
#         w = 0
#         print('(%d, '%s', '%s', '%f', '%f', '%f', '', -1, %d),' % (k, i, reg['GalaxyName'], reg['RA'], reg['DEC'], w, m))
#         k = k + 1
# print(';')
