#!/usr/bin/python3
import os
import sys
import pickle
import numpy as np
from pytu.lines import Lines
from pytu.functions import debug_var
from pytu.objects import readFileArgumentParser


def parser_args(default_args_file='args/default_selection.args'):
    """
    Parse the command line args.

    With fromfile_pidxrefix_chars=@ we can read and parse command line args
    inside a file with @file.txt.
    default args inside default_args_file
    """
    default_args = {
        'input': 'dataframes.pkl',
        'output': 'elines_clean.pkl',
        'broad_fit_rules': False,
        'no_sigma_clip': False,
        'print_color': False,
        'csv_dir': 'csv',
        'bug': 0.8,
        'output_agn_candidates': 'AGN_CANDIDATES.csv',
        'EW_SF': 10.,
        'EW_AGN': 3.,
        'EW_hDIG': 3.,
        'EW_strong': 6.,
        'EW_verystrong': 10.,
        'min_SN_broad': 8.,
        'only_report': False,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input', '-I', metavar='FILE', type=str, default=default_args['input'])
    parser.add_argument('--output', '-O', metavar='FILE', type=str, default=default_args['output'])
    parser.add_argument('--broad_fit_rules', '-B', action='store_true', default=default_args['broad_fit_rules'])
    parser.add_argument('--only_report', '-R', action='store_true', default=default_args['only_report'])
    parser.add_argument('--print_color', action='store_true', default=default_args['print_color'])
    parser.add_argument('--no_sigma_clip', action='store_true', default=default_args['no_sigma_clip'])
    parser.add_argument('--csv_dir', metavar='DIR', type=str, default=default_args['csv_dir'])
    parser.add_argument('--output_agn_candidates', metavar='FILE', type=str, default=default_args['output_agn_candidates'])
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--EW_SF', metavar='FLOAT', type=float, default=default_args['EW_SF'])
    parser.add_argument('--EW_AGN', metavar='FLOAT', type=float, default=default_args['EW_AGN'])
    parser.add_argument('--EW_hDIG', metavar='FLOAT', type=float, default=default_args['EW_hDIG'])
    parser.add_argument('--EW_strong', metavar='FLOAT', type=float, default=default_args['EW_strong'])
    parser.add_argument('--EW_verystrong', metavar='FLOAT', type=float, default=default_args['EW_verystrong'])
    parser.add_argument('--min_SN_broad', metavar='FLOAT', type=float, default=default_args['min_SN_broad'])
    parser.add_argument('--bug', metavar='FLOAT', type=float, default=default_args['bug'])
    args_list = sys.argv[1:]
    # if exists file default.args, load default args
    if os.path.isfile(default_args_file):
        args_list.insert(0, '@%s' % default_args_file)
    debug_var(True, args_list=args_list)
    args = parser.parse_args(args=args_list)
    args = parser.parse_args(args=args_list)
    debug_var(True, args=args)
    return args


def fBPT(x, a, b, c):
    return a + (b/(x + c))


def linval(x, a, b):
    return np.polyval([a, b], x)


def morph_adjust(x):
    r = x
    # If not NAN or M_TYPE E* (0-7) call it E
    if ~np.isnan(x) and x <= 7:
        r = 7
    return r


if __name__ == '__main__':
    args = parser_args()

    if args.print_color:
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

    with open(args.input, 'rb') as f:
        df = pickle.load(f)
    elines = df['elines']

    # REMOVE SOME GALS FROM AGN STUDY
    with open('%s/QC_Pipe3D_CALIFA.csv' % args.csv_dir, 'r') as f:
        DBNames_to_drop = []
        for l in f.readlines():
            if l[0] != '#':
                tmp = l.split(',')
                DBName = tmp[0].strip()
                flag = int(tmp[1])
                if flag > 0:
                    if DBName in df['elines'].index:
                        DBNames_to_drop.append(DBName)
                        print('%s%s%s: removing galaxy from analisys' % (color.B, DBName, color.E))
                    # else:
                    #     print('%s: not in %s' % (DBName, fnames_long['elines']))
        if len(DBNames_to_drop) > 0:
            df_elines_clean = df['elines'].drop(DBNames_to_drop, axis=0)
            elines = df_elines_clean.copy()
            del df_elines_clean
    del df

    # print_str = '{}\t{}\t{}\t{}\t{}\t{}\t{}'
    # # Create file for R. Callete
    # print(print_str.format('DBName', 'redshift', 'Mabs_R', 'u', 'g', 'r', 'i'))
    # for i in elines.index:
    #     t = elines.loc[i]
    #     print(print_str.format(i, t['z_stars'], t['Mabs_R'], t['u'], t['g'], t['r'], t['i']))

    # log_NII_Ha_cen = elines['log_NII_Ha_cen_mean']
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
    eEW_Ha_Re = elines['e_EW_Ha_Re']
    EW_Ha_ALL = elines['EW_Ha_ALL']
    eEW_Ha_ALL = elines['e_EW_Ha_ALL']
    ###############################################################
    L = Lines()
    # consts_K01 = L.consts['K01']
    # consts_K01_SII_Ha = L.consts['K01_SII_Ha']
    # consts_K01_OI_Ha = L.consts['K01_OI_Ha']
    # consts_K03 = L.consts['K03']
    # consts_S06 = L.consts['S06']
    # consts_K06_SII_Ha = L.consts['K06_SII_Ha']
    # consts_K06_OI_Ha = L.consts['K06_OI_Ha']
    if args.no_sigma_clip is False:
        sigma_clip = True
        # consts_K01 = L.sigma_clip_consts['K01']
        # consts_K01_SII_Ha = L.sigma_clip_consts['K01_SII_Ha']
        ###############################################################
    ###############################################################
    # [OIII] vs [NII]
    ###############################################################
    # AGN/LINER
    # y_mod_K01 = log_NII_Ha_cen.apply(fBPT, args=consts_K01)
    # y_mod_K03 = log_NII_Ha_cen.apply(fBPT, args=consts_K03)
    # y_mod_S06 = log_NII_Ha_cen.apply(fBPT, args=consts_S06)
    ###############################################################
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII]
    ###############################################################
    # AGN
    # y_mod_K01_SII = log_SII_Ha_cen.apply(fBPT, args=consts_K01_SII_Ha)
    # y_mod_K06_SII = log_SII_Ha_cen.apply(linval, args=consts_K06_SII_Ha)
    ###############################################################
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [OI]
    ###############################################################
    # y_mod_K01_OI = log_OI_Ha_cen.apply(fBPT, args=consts_K01_OI_Ha)
    # y_mod_K06_OI = log_OI_Ha_cen.apply(linval, args=consts_K06_OI_Ha)
    ###############################################################
    ###############################################################
    # SELECTIONS
    ###############################################################
    ###############################################################
    sel_NIIHa = ~(log_NII_Ha_cen.apply(np.isnan))
    sel_OIIIHb = ~(log_OIII_Hb_cen.apply(np.isnan))
    sel_SIIHa = ~(log_SII_Ha_cen.apply(np.isnan))
    sel_OIHa = ~(log_OI_Ha_cen.apply(np.isnan))
    sel_MS = sel_NIIHa & sel_OIIIHb & sel_SIIHa & sel_OIHa
    sel_EW_cen = ~(EW_Ha_cen.apply(np.isnan))
    sel_EW_Re = ~(EW_Ha_Re.apply(np.isnan))
    sel_EW_ALL = ~(EW_Ha_ALL.apply(np.isnan))
    # K01
    sel_below_K01 = L.belowlinebpt('K01', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_below_K01_SII = L.belowlinebpt('K01_SII_Ha', log_SII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_below_K01_OI = L.belowlinebpt('K01_OI_Ha', log_OI_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    # K03
    sel_below_K03 = L.belowlinebpt('K03', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    # S06
    sel_below_S06 = L.belowlinebpt('S06', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    # K06
    sel_below_CF10 = L.belowlinebpt('CF10', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_below_K06_SII = L.belowlinebpt('K06_SII_Ha', log_SII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_below_K06_OI = L.belowlinebpt('K06_OI_Ha', log_OI_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)

    sel_AGNLINER_NIIHa_OIIIHb = sel_NIIHa & sel_OIIIHb & ~sel_below_K01
    sel_AGN_NIIHa_OIIIHb_K01_CF10 = sel_NIIHa & sel_OIIIHb & ~sel_below_CF10 & ~sel_below_K01
    sel_AGN_SIIHa_OIIIHb_K01 = sel_SIIHa & sel_OIIIHb & ~sel_below_K01_SII
    sel_AGN_OIHa_OIIIHb_K01 = sel_OIHa & sel_OIIIHb & ~sel_below_K01_OI
    sel_AGN_SIIHa_OIIIHb_K01_K06 = sel_SIIHa & sel_OIIIHb & ~sel_below_K01_SII & ~sel_below_K06_SII
    sel_AGN_OIHa_OIIIHb_K01_K06 = sel_OIHa & sel_OIIIHb & ~sel_below_K01_OI & ~sel_below_K06_OI
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01 & sel_AGN_OIHa_OIIIHb_K01
    sel_AGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug))
    sel_SAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug))
    sel_VSAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug))
    sel_SF_NIIHa_OIIIHb_K01 = sel_NIIHa & sel_OIIIHb & sel_below_K01
    sel_SF_NIIHa_OIIIHb_K03 = sel_NIIHa & sel_OIIIHb & L.belowlinebpt('K03', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_SF_NIIHa_OIIIHb_S06 = sel_NIIHa & sel_OIIIHb & L.belowlinebpt('S06', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_pAGB = sel_EW_cen & (EW_Ha_cen <= args.EW_hDIG*args.bug)
    sel_SF_EW = sel_EW_cen & (EW_Ha_cen > args.EW_SF*args.bug)
    sel_SF_EW_CanoDiaz = sel_EW_cen & (EW_Ha_cen > 6*args.bug)

    # MAIN SAMPLE SELECTIONS (EXCLUDING POINTS WITHOUT SII AND OI)
    sel_AGNLINER_NIIHa_OIIIHb_MS = sel_MS & ~sel_below_K01
    sel_AGN_NIIHa_OIIIHb_K01_CF10_MS = sel_MS & ~sel_below_CF10 & ~sel_below_K01
    sel_AGN_SIIHa_OIIIHb_K01_MS = sel_MS & ~sel_below_K01_SII
    sel_AGN_OIHa_OIIIHb_K01_MS = sel_MS & ~sel_below_K01_OI
    sel_AGN_SIIHa_OIIIHb_K01_K06_MS = sel_MS & ~sel_below_K01_SII & ~sel_below_K06_SII
    sel_AGN_OIHa_OIIIHb_K01_K06_MS = sel_MS & ~sel_below_K01_OI & ~sel_below_K06_OI
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS = sel_AGNLINER_NIIHa_OIIIHb_MS & sel_AGN_SIIHa_OIIIHb_K01_MS
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS = sel_AGNLINER_NIIHa_OIIIHb_MS & sel_AGN_OIHa_OIIIHb_K01_MS
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS = sel_AGNLINER_NIIHa_OIIIHb_MS & sel_AGN_SIIHa_OIIIHb_K01_MS & sel_AGN_OIHa_OIIIHb_K01_MS
    sel_AGN_candidates_MS = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug))
    sel_SAGN_candidates_MS = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug))
    sel_VSAGN_candidates_MS = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug))
    sel_SF_NIIHa_OIIIHb_K01_MS = sel_MS & sel_below_K01
    sel_SF_NIIHa_OIIIHb_K03_MS = sel_MS & L.belowlinebpt('K03', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_SF_NIIHa_OIIIHb_S06_MS = sel_MS & L.belowlinebpt('S06', log_NII_Ha_cen, log_OIII_Hb_cen, sigma_clip=sigma_clip)
    sel_pAGB_MS = sel_MS & sel_EW_cen & (EW_Ha_cen <= args.EW_hDIG*args.bug)
    sel_SF_EW_MS = sel_MS & sel_EW_cen & (EW_Ha_cen > args.EW_SF*args.bug)

    ###############################################################################
    # END SETTING VALUES ##########################################################
    ###############################################################################

    ###############################################################################
    # BEGIN REPORTS ###############################################################
    ###############################################################################
    ###############################################################################
    # BEGIN REPORT RATIOS #########################################################
    ###############################################################################
    print('\n#RR#################')
    print('#RR# REPORT RATIOS #')
    print('#RR#################')
    groups = [
        ['log_NII_Ha_cen'],
        ['log_SII_Ha_cen_mean'],
        ['log_OI_Ha_cen'],
        ['log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen', 'log_SII_Ha_cen_mean', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
        ['log_NII_Ha_cen', 'log_SII_Ha_cen_mean', 'log_OI_Ha_cen', 'log_OIII_Hb_cen_mean'],
    ]
    for g in groups:
        if len(g) > 1:
            N = elines.groupby(g).ngroups
        else:
            N = elines[g[0]].count()
        print('#RR# %s measured: %d galaxies' % (g, N))
    print('#RR#################\n')
    ###############################################################################
    # END REPORT RATIOS ###########################################################
    ###############################################################################

    ###############################################################################
    # BEGIN REPORT AGN CANDIDATES #################################################
    ###############################################################################
    print('\n#AC##################')
    print('#AC# AGN CANDIDATES #')
    N_TOT = len(elines.index)
    N_GAS = len(elines.loc[elines['F_Ha_cen'] > 0])
    N_NO_GAS = N_TOT - N_GAS
    N_GAS_BPT = len(elines.loc[~(elines['log_NII_Ha_cen'].apply(np.isnan)) & ~(elines['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
    N_NO_GAS_BPT = N_TOT - N_GAS_BPT
    N_GAS_MASS_ESTIM = len(elines.loc[~elines['log_Mass_gas_Av_gas_rad'].apply(np.isnan)])
    ###############################################################
    # [OIII] vs [NII]
    ###############################################################
    # AGN/LINER
    ###############################################################
    # m = sel_AGNLINER_NIIHa_OIIIHb_MS
    m = sel_AGNLINER_NIIHa_OIIIHb
    N_AGN_NII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug)).values.astype('int').sum()
    N_SAGN_NII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    # elines.loc[(m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)), 'TYPE'] = 7
    N_VSAGN_NII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    # SF
    # m = sel_SF_NIIHa_OIIIHb_K01_MS
    m = sel_SF_NIIHa_OIIIHb_K01
    N_SF_K01 = m.values.astype('int').sum()
    N_SF_K01_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SSF_K01 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSSF_K01 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    # m = sel_SF_NIIHa_OIIIHb_K03_MS
    m = sel_SF_NIIHa_OIIIHb_K03
    N_SF_K03 = m.values.astype('int').sum()
    N_SF_K03_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW_cen & (EW_Ha_cen > args.EW_hDIG*args.bug)), 'TYPE'] = 6
    N_SSF_K03 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)), 'TYPE'] = 1
    N_VSSF_K03 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    # m = sel_SF_NIIHa_OIIIHb_S06_MS
    m = sel_SF_NIIHa_OIIIHb_S06
    N_SF_S06 = m.values.astype('int').sum()
    N_SF_S06_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SSF_S06 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSSF_S06 = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [SII]
    ###############################################################
    # m = sel_AGN_SIIHa_OIIIHb_K01_MS
    m = sel_AGN_SIIHa_OIIIHb_K01
    N_AGN_SII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug)).values.astype('int').sum()
    N_SAGN_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII]
    ###############################################################
    # m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS
    m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01
    N_AGN_NII_SII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug)).values.astype('int').sum()
    N_SAGN_NII_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_NII_SII_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [OI]
    ###############################################################
    # m = sel_AGN_OIHa_OIIIHb_K01_MS
    m = sel_AGN_OIHa_OIIIHb_K01
    N_AGN_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug)).values.astype('int').sum()
    N_SAGN_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [OI]
    ###############################################################
    # m = sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS
    m = sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01
    N_AGN_NII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug)).values.astype('int').sum()
    N_SAGN_NII_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_NII_OI_Ha_EW = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII] + [OIII] vs [OI]
    ###############################################################
    ###############################################################
    # m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS
    m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01
    N_AGN_NII_SII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    sel_AGN = (m & sel_EW_cen & (EW_Ha_cen > args.EW_AGN*args.bug))
    N_AGN = sel_AGN.values.astype('int').sum()
    elines.loc[sel_AGN, 'TYPE'] = 2
    sel_SAGN = (m & sel_EW_cen & (EW_Ha_cen > args.EW_strong*args.bug))
    N_SAGN = sel_SAGN.values.astype('int').sum()
    elines.loc[sel_SAGN, 'TYPE'] = 3
    sel_VSAGN = (m & sel_EW_cen & (EW_Ha_cen > args.EW_verystrong*args.bug))
    N_VSAGN = sel_VSAGN.values.astype('int').sum()
    ###############################################################
    ###############################################################
    # pAGB
    ##############################################################
    # m = sel_pAGB_MS
    m = sel_pAGB
    N_pAGB = m.values.astype('int').sum()
    elines.loc[m, 'TYPE'] = 4
    N_pAGB_below_K01 = (m & sel_NIIHa & sel_OIIIHb & sel_below_K01).values.astype('int').sum()
    N_pAGB_below_K03 = (m & sel_NIIHa & sel_OIIIHb & sel_below_K03).values.astype('int').sum()
    N_pAGB_below_S06 = (m & sel_NIIHa & sel_OIIIHb & sel_below_S06).values.astype('int').sum()
    N_pAGB_below_K01_SII = (m & sel_SIIHa & sel_OIIIHb & sel_below_K01_SII).values.astype('int').sum()
    N_pAGB_below_K01_OI = (m & sel_OIHa & sel_OIIIHb & sel_below_K01_OI).values.astype('int').sum()
    ###############################################################
    ###############################################################
    # SF
    ###############################################################
    # m = sel_SF_EW_MS
    m = sel_SF_EW
    N_SF_EW = m.values.astype('int').sum()
    N_SF_EW_above_K01 = (m & sel_NIIHa & sel_OIIIHb & ~sel_below_K01).values.astype('int').sum()
    N_SF_EW_above_K03 = (m & sel_NIIHa & sel_OIIIHb & ~sel_below_K03).values.astype('int').sum()
    N_SF_EW_above_S06 = (m & sel_NIIHa & sel_OIIIHb & ~sel_below_S06).values.astype('int').sum()
    N_SF_EW_above_K01_SII = (m & sel_SIIHa & sel_OIIIHb & ~sel_below_K01_SII).values.astype('int').sum()
    N_SF_EW_above_K01_OI = (m & sel_OIHa & sel_OIIIHb & ~sel_below_K01_OI).values.astype('int').sum()
    ###############################################################
    # m = (EW_Ha_cen > args.EW_strong*args.bug) & (sel_AGNLINER_NIIHa_OIIIHb | sel_AGN_SIIHa_OIIIHb_K01 | sel_AGN_OIHa_OIIIHb_K01)
    # m = (EW_Ha_cen > args.EW_AGN*args.bug) & sel_AGNLINER_NIIHa_OIIIHb_MS
    m = (EW_Ha_cen > args.EW_AGN*args.bug) & sel_AGNLINER_NIIHa_OIIIHb
    elines.loc[m, 'AGN_FLAG'] = 4
    # m = (EW_Ha_cen > args.EW_strong*args.bug) & ((sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01) | (sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01) | (sel_AGN_SIIHa_OIIIHb_K01 & sel_AGN_OIHa_OIIIHb_K01))
    # m = (EW_Ha_cen > args.EW_AGN*args.bug) & ((sel_AGNLINER_NIIHa_OIIIHb_MS & sel_AGN_SIIHa_OIIIHb_K01_MS) | (sel_AGNLINER_NIIHa_OIIIHb_MS & sel_AGN_OIHa_OIIIHb_K01_MS))
    m = (EW_Ha_cen > args.EW_AGN*args.bug) & ((sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01) | (sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01))
    elines.loc[m, 'AGN_FLAG'] = 3
    m = ((elines['TYPE'] == 2) | (elines['TYPE'] == 3))
    elines.loc[m, 'AGN_FLAG'] = 2
    # m = ((elines['SN_broad'] > 8) & ((elines['TYPE'] == 2) | (elines['TYPE'] == 3))) | (elines['broad_by_eye'] == True)
    m = (elines['SN_broad'] > args.min_SN_broad) & (elines['AGN_FLAG'] == 2) | (elines['broad_by_eye'] == True)
    if args.broad_fit_rules is True:
        m = (elines['SN_broad'] > args.min_SN_broad)
    elines.loc[m, 'AGN_FLAG'] = 1
    # Correct NII_Ha for broad type AGNs
    m = (elines['AGN_FLAG'] == 1)
    elines['log_NII_Ha_cen'] = np.where(m, elines['log_NII_Ha_cen_fit'], elines['log_NII_Ha_cen_mean'])

    N_AGN_tI = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 1].count()
    N_AGN_tII = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 2].count()
    N_AGN_tIII = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 3].count()
    N_AGNLINER_N2Ha = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 4].count()

    # OUTPUT ######################################################################
    print('#AC##################')
    print('#AC# %sN.TOTAL%s = %d' % (color.B, color.E, N_TOT))
    print('#AC# %sN.GAS MASS ESTIMATED%s = %d' % (color.B, color.E, N_GAS_MASS_ESTIM))
    print('#AC# %sN.NO GAS%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS_BPT))
    print('#AC# %sN.NO GAS%s (without %sF_Ha_cen%s) = %d' % (color.B, color.E, color.B, color.E, N_NO_GAS))
    print('#AC##################')
    print('#AC# %sEW cuts%s:' % (color.B, color.E))
    print('#AC# \t%snot-pAGB%s (%sN%s): EW > %.2f * %.2f = %.2f A' % (color.B, color.E, color.B, color.E, args.EW_hDIG, args.bug, args.EW_hDIG*args.bug))
    print('#AC# \t%sAGN%s (%sA%s): EW > %.2f * %.2f = %.2f A' % (color.B, color.E, color.B, color.E, args.EW_AGN, args.bug, args.EW_AGN*args.bug))
    print('#AC# \t%sStrong%s (%sS%s): EW > %.2f * %.2f = %.2f A' % (color.B, color.E, color.B, color.E, args.EW_strong, args.bug, args.EW_strong*args.bug))
    print('#AC# \t%sVery strong%s (%sVS%s): EW > %.2f * %.2f = %.2f A' % (color.B, color.E, color.B, color.E, args.EW_verystrong, args.bug, args.EW_verystrong*args.bug))
    print('#AC# \t%sSF%s: EW > %.2f * %.2f = %.2f A' % (color.B, color.E, args.EW_SF, args.bug, args.EW_SF*args.bug))
    print('#AC##################')
    print('#AC# N.AGNs/LINERs candidates by [NII]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_Ha, color.B, color.E, N_AGN_NII_Ha_EW, color.B, color.E, N_SAGN_NII_Ha_EW, color.B, color.E, N_VSAGN_NII_Ha_EW))
    print('#AC# N.AGNs candidates by [SII]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_SII_Ha, color.B, color.E, N_AGN_SII_Ha_EW, color.B, color.E, N_SAGN_SII_Ha_EW, color.B, color.E, N_VSAGN_SII_Ha_EW))
    print('#AC# N.AGNs candidates by [OI]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_OI_Ha, color.B, color.E, N_AGN_OI_Ha_EW, color.B, color.E, N_SAGN_OI_Ha_EW, color.B, color.E, N_VSAGN_OI_Ha_EW))
    print('#AC# N.AGNs candidates by [NII]/Ha and [SII]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_Ha, color.B, color.E, N_AGN_NII_SII_Ha_EW, color.B, color.E, N_SAGN_NII_SII_Ha_EW, color.B, color.E, N_VSAGN_NII_SII_Ha_EW))
    print('#AC# N.AGNs candidates by [NII]/Ha and [OI]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_OI_Ha, color.B, color.E, N_AGN_NII_OI_Ha_EW, color.B, color.E, N_SAGN_NII_OI_Ha_EW, color.B, color.E, N_VSAGN_NII_OI_Ha_EW))
    print('#AC# N.AGNs candidates by [NII]/Ha, [SII]/Ha and [OI]/Ha: %d (%sA%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_OI_Ha, color.B, color.E, N_AGN, color.B, color.E, N_SAGN, color.B, color.E, N_VSAGN))
    print('#AC# N.AGNs %sAGN/LINER%s: %d - %sAGN N2Ha+%s: %d - %sType-II%s: %d - %sType-I%s: %d' % (color.B, color.E, N_AGNLINER_N2Ha, color.B, color.E, N_AGN_tIII, color.B, color.E, N_AGN_tII, color.B, color.E, N_AGN_tI))
    print('#AC# N.pAGB: %d (%sbelow K01%s: %d - %sbelow K03%s: %d - %sbelow S06%s: %d - %sbelow K01 SII%s: %d - %sbelow K01 OI%s: %d)' % (N_pAGB, color.B, color.E, N_pAGB_below_K01, color.B, color.E, N_pAGB_below_K03, color.B, color.E, N_pAGB_below_S06, color.B, color.E, N_pAGB_below_K01_SII, color.B, color.E, N_pAGB_below_K01_OI))
    print('#AC# N_SF %sEW%s: %d (%sabove K01%s: %d - %sabove K03%s: %d - %sabove S06%s: %d - %sabove K01_SII%s: %d - %sabove K01_OI%s: %d)' % (color.B, color.E, N_SF_EW, color.B, color.E, N_SF_EW_above_K01, color.B, color.E, N_SF_EW_above_K03, color.B, color.E, N_SF_EW_above_S06, color.B, color.E, N_SF_EW_above_K01_SII, color.B, color.E, N_SF_EW_above_K01_OI))
    print('#AC# N.SF %sK01%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K01, color.B, color.E, N_SF_K01_EW, color.B, color.E, N_SSF_K01, color.B, color.E, N_VSSF_K01))
    print('#AC# N.SF %sK03%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K03, color.B, color.E, N_SF_K03_EW, color.B, color.E, N_SSF_K03, color.B, color.E, N_VSSF_K03))
    print('#AC# N.SF %sS06%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_S06, color.B, color.E, N_SF_S06_EW, color.B, color.E, N_SSF_S06, color.B, color.E, N_VSSF_S06))
    print('#AC##################\n')

    morph_name = ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'I']
    elines_wmorph = elines.loc[elines['morph'] >= 0]
    N_TOT_WMORPH = len(elines_wmorph.index)
    N_GAS_WMORPH = len(elines_wmorph.loc[elines_wmorph['F_Ha_cen'] > 0])
    N_NO_GAS_WMORPH = N_TOT_WMORPH - N_GAS_WMORPH
    N_GAS_BPT_WMORPH = len(elines_wmorph.loc[~(elines_wmorph['log_NII_Ha_cen'].apply(np.isnan)) & ~(elines_wmorph['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
    N_NO_GAS_BPT_WMORPH = N_TOT_WMORPH - N_GAS_BPT_WMORPH
    N_GAS_MASS_ESTIM_WMORPH = len(elines_wmorph.loc[~elines_wmorph['log_Mass_gas_Av_gas_rad'].apply(np.isnan)])

    print('###################')
    print('## Morph studies ##')
    print('###################\n')
    print('#AC##################')
    print('#AC# %sN.TOTAL%s = %d' % (color.B, color.E, N_TOT_WMORPH))
    print('#AC# %sN.GAS MASS ESTIMATED%s = %d' % (color.B, color.E, N_GAS_MASS_ESTIM_WMORPH))
    print('#AC# %sN.NO GAS%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS_BPT_WMORPH))
    print('#AC# %sN.NO GAS%s (without %sF_Ha_cen%s) = %d' % (color.B, color.E, color.B, color.E, N_NO_GAS_WMORPH))
    print('#AC##################\n')
    ###############################################################################
    # END REPORTS #################################################################
    ###############################################################################

    if not args.only_report:
        columns_to_csv = [
            'AGN_FLAG', 'SN_broad',
            'RA', 'DEC', 'log_NII_Ha_cen', 'log_NII_Ha_cen_stddev',
            'log_OIII_Hb_cen_mean', 'log_OIII_Hb_cen_stddev',
            'log_SII_Ha_cen_mean', 'log_SII_Ha_cen_stddev',
            'log_OI_Ha_cen', 'e_log_OI_Ha_cen',
            'EW_Ha_cen_mean', 'EW_Ha_cen_stddev',
        ]
        m = elines['AGN_FLAG'] > 0
        elines.loc[m].sort_values('AGN_FLAG').to_csv('%s/%s' % (args.csv_dir, args.output_agn_candidates), columns=columns_to_csv)

        # m = (elines['AGN_FLAG'] == 1) | (elines['AGN_FLAG'] == 2)
        # print(elines.loc[m, columns_to_csv].sort_values('AGN_FLAG'))

        to_save = {
            'df': elines,
            'bug': args.bug,
            'EW_SF': args.EW_SF,
            'EW_AGN': args.EW_AGN,
            'EW_hDIG': args.EW_hDIG,
            'EW_strong': args.EW_strong,
            'EW_verystrong': args.EW_verystrong,
            'sel_NIIHa': sel_NIIHa,
            'sel_OIIIHb': sel_OIIIHb,
            'sel_SIIHa': sel_SIIHa,
            'sel_OIHa': sel_OIHa,
            'sel_MS': sel_MS,
            'sel_EW_cen': sel_EW_cen,
            'sel_EW_ALL': sel_EW_ALL,
            'sel_EW_Re': sel_EW_Re,
            'sel_below_K01': sel_below_K01,
            'sel_below_K01_SII': sel_below_K01_SII,
            'sel_below_K01_OI': sel_below_K01_OI,
            'sel_below_K03': sel_below_K03,
            'sel_below_S06': sel_below_S06,
            'sel_below_CF10': sel_below_CF10,
            'sel_below_K06_SII': sel_below_K06_SII,
            'sel_below_K06_OI': sel_below_K06_OI,
            'sel_AGNLINER_NIIHa_OIIIHb': sel_AGNLINER_NIIHa_OIIIHb,
            'sel_AGN_NIIHa_OIIIHb_K01_CF10': sel_AGN_NIIHa_OIIIHb_K01_CF10,
            'sel_AGN_SIIHa_OIIIHb_K01': sel_AGN_SIIHa_OIIIHb_K01,
            'sel_AGN_OIHa_OIIIHb_K01': sel_AGN_OIHa_OIIIHb_K01,
            'sel_AGN_SIIHa_OIIIHb_K01_K06': sel_AGN_SIIHa_OIIIHb_K01_K06,
            'sel_AGN_OIHa_OIIIHb_K01_K06': sel_AGN_OIHa_OIIIHb_K01_K06,
            'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01,
            'sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01,
            'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01,
            'sel_AGN_candidates': sel_AGN_candidates,
            'sel_SAGN_candidates': sel_SAGN_candidates,
            'sel_VSAGN_candidates': sel_VSAGN_candidates,
            'sel_SF_NIIHa_OIIIHb_K01': sel_SF_NIIHa_OIIIHb_K01,
            'sel_SF_NIIHa_OIIIHb_K03': sel_SF_NIIHa_OIIIHb_K03,
            'sel_SF_NIIHa_OIIIHb_S06': sel_SF_NIIHa_OIIIHb_S06,
            'sel_pAGB': sel_pAGB,
            'sel_SF_EW': sel_SF_EW,
            'sel_AGNLINER_NIIHa_OIIIHb_MS': sel_AGNLINER_NIIHa_OIIIHb_MS,
            'sel_AGN_NIIHa_OIIIHb_K01_CF10_MS': sel_AGN_NIIHa_OIIIHb_K01_CF10_MS,
            'sel_AGN_SIIHa_OIIIHb_K01_MS': sel_AGN_SIIHa_OIIIHb_K01_MS,
            'sel_AGN_OIHa_OIIIHb_K01_MS': sel_AGN_OIHa_OIIIHb_K01_MS,
            'sel_AGN_SIIHa_OIIIHb_K01_K06_MS': sel_AGN_SIIHa_OIIIHb_K01_K06_MS,
            'sel_AGN_OIHa_OIIIHb_K01_K06_MS': sel_AGN_OIHa_OIIIHb_K01_K06_MS,
            'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_MS,
            'sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS': sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01_MS,
            'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01_MS,
            'sel_AGN_candidates_MS': sel_AGN_candidates_MS,
            'sel_SAGN_candidates_MS': sel_SAGN_candidates_MS,
            'sel_VSAGN_candidates_MS': sel_VSAGN_candidates_MS,
            'sel_SF_NIIHa_OIIIHb_K01_MS': sel_SF_NIIHa_OIIIHb_K01_MS,
            'sel_SF_NIIHa_OIIIHb_K03_MS': sel_SF_NIIHa_OIIIHb_K03_MS,
            'sel_SF_NIIHa_OIIIHb_S06_MS': sel_SF_NIIHa_OIIIHb_S06_MS,
            'sel_pAGB_MS': sel_pAGB_MS,
            'sel_SF_EW_MS': sel_SF_EW_MS,
        }

        with open(args.output, 'wb') as f:
            pickle.dump(to_save, f, protocol=2)

    ###############################################################################
    # BEGIN PAPER REPORTS #########################################################
    ###############################################################################
    mtI = elines['AGN_FLAG'] == 1
    N_AGN_tI = mtI.astype('int').sum()
    mtII = elines['AGN_FLAG'] == 2
    N_AGN_tII = mtII.astype('int').sum()
    mBFAGN = mtI | mtII
    N_BFAGN = mBFAGN.astype('int').sum()
    mAGN = elines['AGN_FLAG'] > 0
    N_ALLAGN = mAGN.astype('int').sum()
    x = elines['morph'].apply(morph_adjust)
    H, _ = np.histogram(x, bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtI, _ = np.histogram(x[mtI], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HtII, _ = np.histogram(x[mtII], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HAGN, _ = np.histogram(x[mAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    HBFAGN, _ = np.histogram(x[mBFAGN], bins=np.linspace(6.5, 19.5, 14), range=[6.5, 19.5])
    print('mtp\ttot\ttI\ttII\ttBFAGN\t\ttAGN')
    for mtyp, tot, tI, tII, tBFAGN, tAGN in zip(morph_name[7:], H, HtI, HtII, HBFAGN, HAGN):
        print('%s\t%d\t%d\t%d\t%d\t\t%d' % (mtyp, tot, tI, tII, tBFAGN, tAGN))
    N_elipt = H[0]
    N_lent = H[1] + H[2]
    N_ET = N_elipt + N_lent
    N_spirals = H[3:-1].sum()
    N_irr = H[-1]
    N_LT = N_spirals + N_irr
    print('N_elipt:{}\tN_lent:{}\tN_ET:{}\tN_spirals:{}\tN_irr:{}\tN_LT:{}'.format(N_elipt, N_lent, N_ET, N_spirals, N_irr, N_LT))

    N_SF_EW_NIIHa_OIIIHb = (sel_NIIHa & sel_OIIIHb & sel_SF_EW).astype('int').sum()
    N_SF_EW_SIIHa_OIIIHb = (sel_SIIHa & sel_OIIIHb & sel_SF_EW).astype('int').sum()
    N_SF_EW_OIHa_OIIIHb = (sel_OIHa & sel_OIIIHb & sel_SF_EW).astype('int').sum()
    N_SFCD_EW_NIIHa_OIIIHb = (sel_NIIHa & sel_OIIIHb & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_SFCD_EW_SIIHa_OIIIHb = (sel_SIIHa & sel_OIIIHb & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_SFCD_EW_OIHa_OIIIHb = (sel_OIHa & sel_OIIIHb & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_pAGB_NIIHa_OIIIHb = (sel_NIIHa & sel_OIIIHb & sel_pAGB).astype('int').sum()
    N_pAGB_SIIHa_OIIIHb = (sel_SIIHa & sel_OIIIHb & sel_pAGB).astype('int').sum()
    N_pAGB_OIHa_OIIIHb = (sel_OIHa & sel_OIIIHb & sel_pAGB).astype('int').sum()
    print('N_SF_EW_NIIHa_OIIIHb:{}  N_SF_EW_SIIHa_OIIIHb:{}  N_SF_EW_OIHa_OIIIHb:{}  N_SFCD_EW_NIIHa_OIIIHb:{}  N_SFCD_EW_SIIHa_OIIIHb:{}  N_SFCD_EW_OIHa_OIIIHb:{}  N_pAGB_NIIHa_OIIIHb:{}  N_pAGB_SIIHa_OIIIHb:{}  N_pAGB_OIHa_OIIIHb:{}  '.format(N_SF_EW_NIIHa_OIIIHb,N_SF_EW_SIIHa_OIIIHb,N_SF_EW_OIHa_OIIIHb,N_SFCD_EW_NIIHa_OIIIHb,N_SFCD_EW_SIIHa_OIIIHb,N_SFCD_EW_OIHa_OIIIHb,N_pAGB_NIIHa_OIIIHb,N_pAGB_SIIHa_OIIIHb,N_pAGB_OIHa_OIIIHb))

    m = ~sel_below_K01
    N_above_K01 = m.astype('int').sum()
    N_pAGB_above_K01 = (m & sel_pAGB).astype('int').sum()
    N_SF_EW_above_K01 = (m & sel_SF_EW).astype('int').sum()
    N_SFCD_EW_above_K01 = (m & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_mtI_above_K01 = (m & mtI).astype('int').sum()
    N_mtII_above_K01 = (m & mtII).astype('int').sum()
    N_AGN_SF_above_K01 = (m & sel_SF_EW & (mtI | mtII)).astype('int').sum()
    print('N_above_K01:{}  pAGB:{}  SF:{}  SFCD:{}  N_tI:{}  N_tII:{}  N_AGN_SF_above_K01:{}'.format(N_above_K01, N_pAGB_above_K01, N_SF_EW_above_K01, N_SFCD_EW_above_K01, N_mtI_above_K01, N_mtII_above_K01, N_AGN_SF_above_K01))

    m = ~sel_below_K01_SII
    N_above_K01_SII = m.astype('int').sum()
    N_pAGB_above_K01_SII = (m & sel_pAGB).astype('int').sum()
    N_SF_EW_above_K01_SII = (m & sel_SF_EW).astype('int').sum()
    N_SFCD_EW_above_K01_SII = (m & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_mtI_above_K01_SII = (m & mtI).astype('int').sum()
    N_mtII_above_K01_SII = (m & mtII).astype('int').sum()
    N_AGN_SF_above_K01_SII = (m & sel_SF_EW & (mtI | mtII)).astype('int').sum()
    print('N_above_K01_SII:{}  pAGB:{}  SF:{}  SFCD:{}  N_tI:{}  N_tII:{}  N_AGN_SF_above_K01_SII:{}'.format(N_above_K01_SII, N_pAGB_above_K01_SII, N_SF_EW_above_K01_SII, N_SF_EW_above_K01_SII, N_mtI_above_K01_SII, N_mtII_above_K01_SII, N_AGN_SF_above_K01_SII))

    m = ~sel_below_K01_OI
    N_above_K01_OI = m.astype('int').sum()
    N_pAGB_above_K01_OI = (m & sel_pAGB).astype('int').sum()
    N_SF_EW_above_K01_OI = (m & sel_SF_EW).astype('int').sum()
    N_SFCD_EW_above_K01_OI = (m & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_mtI_above_K01_OI = (m & mtI).astype('int').sum()
    N_mtII_above_K01_OI = (m & mtII).astype('int').sum()
    N_AGN_SF_above_K01_OI = (m & sel_SF_EW & (mtI | mtII)).astype('int').sum()
    print('N_above_K01_OI:{}  pAGB:{}  SF:{}  SFCD:{}  N_tI:{}  N_tII:{}  N_AGN_SF_above_K01_OI:{}'.format(N_above_K01_OI, N_pAGB_above_K01_OI, N_SF_EW_above_K01_OI, N_SF_EW_above_K01_OI, N_mtI_above_K01_OI, N_mtII_above_K01_OI, N_AGN_SF_above_K01_OI))

    m = ~sel_below_K03
    N_above_K03 = m.astype('int').sum()
    N_pAGB_above_K03 = (m & sel_pAGB).astype('int').sum()
    N_SF_EW_above_K03 = (m & sel_SF_EW).astype('int').sum()
    N_SFCD_EW_above_K03 = (m & sel_SF_EW_CanoDiaz).astype('int').sum()
    N_mtI_above_K03 = (m & mtI).astype('int').sum()
    N_mtII_above_K03 = (m & mtII).astype('int').sum()
    N_AGN_SF_above_K03 = (m & sel_SF_EW & (mtI | mtII)).astype('int').sum()
    print('N_above_K03:{}  pAGB:{}  SF:{}  SFCD:{}  N_tI:{}  N_tII:{}  N_AGN_SF_above_K01:{}'.format(N_above_K03, N_pAGB_above_K03, N_SF_EW_above_K03, N_SFCD_EW_above_K03, N_mtI_above_K03, N_mtII_above_K03, N_AGN_SF_above_K03))

    N_pAGB_all = (sel_pAGB & sel_OIIIHb & sel_NIIHa & sel_SIIHa & sel_OIHa).astype('int').sum()
    N_pAGB_inBPT = (sel_pAGB & sel_OIIIHb & sel_NIIHa).astype('int').sum()
    N_pAGB_inSII = (sel_pAGB & sel_OIIIHb & sel_SIIHa).astype('int').sum()
    N_pAGB_inOI = (sel_pAGB & sel_OIIIHb & sel_OIHa).astype('int').sum()
    print('N_pAGB_all:{}  N_pAGB_inBPT:{}  N_pAGB_inSII:{}  N_pAGB_inOI:{}'.format(N_pAGB_all, N_pAGB_inBPT, N_pAGB_inSII, N_pAGB_inOI))

    N_SF_EW_all = (sel_SF_EW & sel_OIIIHb & sel_NIIHa & sel_SIIHa & sel_OIHa).astype('int').sum()
    N_SF_EW_inBPT = (sel_SF_EW & sel_OIIIHb & sel_NIIHa).astype('int').sum()
    N_SF_EW_inSII = (sel_SF_EW & sel_OIIIHb & sel_SIIHa).astype('int').sum()
    N_SF_EW_inOI = (sel_SF_EW & sel_OIIIHb & sel_OIHa).astype('int').sum()
    print('N_SF_EW_all:{}  N_SF_EW_inBPT:{}  N_SF_EW_inSII:{}  N_SF_EW_inOI:{}'.format(N_SF_EW_all, N_SF_EW_inBPT, N_SF_EW_inSII, N_SF_EW_inOI))

    N_SFCD_EW_all = (sel_SF_EW_CanoDiaz & sel_OIIIHb & sel_NIIHa & sel_SIIHa & sel_OIHa).astype('int').sum()
    N_SFCD_EW_inBPT = (sel_SF_EW_CanoDiaz & sel_OIIIHb & sel_NIIHa).astype('int').sum()
    N_SFCD_EW_inSII = (sel_SF_EW_CanoDiaz & sel_OIIIHb & sel_SIIHa).astype('int').sum()
    N_SFCD_EW_inOI = (sel_SF_EW_CanoDiaz & sel_OIIIHb & sel_OIHa).astype('int').sum()
    print('N_SFCD_EW_all:{}  N_SFCD_EW_inBPT:{}  N_SFCD_EW_inSII:{}  N_SFCD_EW_inOI:{}'.format(N_SFCD_EW_all, N_SFCD_EW_inBPT, N_SFCD_EW_inSII, N_SFCD_EW_inOI))

    N_AGN_below_K06_SII = (sel_EW_cen & (EW_Ha_cen > 3) & sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_below_K06_SII).astype('int').sum()
    N_AGN_below_K06_OI = (sel_EW_cen & (EW_Ha_cen > 3) & sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_below_K06_OI).astype('int').sum()
    N_AGN_below_CF10 = (sel_below_CF10 & (mtI | mtII)).astype('int').sum()
    print('N_AGN_below_K06_SII:{}  N_AGN_below_K06_OI:{}  N_AGN_below_CF10:{}'.format(N_AGN_below_K06_SII, N_AGN_below_K06_OI, N_AGN_below_CF10))

    ###############################################################################
    # END PAPER REPORTS ###########################################################
    ###############################################################################
