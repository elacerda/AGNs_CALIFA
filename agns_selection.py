#!/home/lacerda/anaconda2/bin/python
import os
import sys
import pickle
import numpy as np
from pytu.lines import Lines
from pytu.functions import debug_var
from pytu.objects import readFileArgumentParser


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


def fBPT(x, a, b, c):
    return a + (b/(x + c))

def linval(x, a, b):
    return np.polyval([a, b], x)


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
        'EW_SF': 14,
        'EW_hDIG': 3,
        'EW_strong': 6,
        'EW_verystrong': 10,
        'min_SN_broad': 8,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input', '-I', metavar='FILE', type=str, default=default_args['input'])
    parser.add_argument('--output', '-O', metavar='FILE', type=str, default=default_args['output'])
    parser.add_argument('--broad_fit_rules', '-B', action='store_true', default=default_args['broad_fit_rules'])
    parser.add_argument('--print_color', action='store_true', default=default_args['print_color'])
    parser.add_argument('--no_sigma_clip', action='store_true', default=default_args['no_sigma_clip'])
    parser.add_argument('--csv_dir', metavar='DIR', type=str, default=default_args['csv_dir'])
    parser.add_argument('--output_agn_candidates', metavar='FILE', type=str, default=default_args['output_agn_candidates'])
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--EW_SF', metavar='FLOAT', type=float, default=default_args['EW_SF'])
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

    # REMOVE SOME GALS FROM AGN STUDY
    with open('%s/remove_gals_AGNpaper.csv' % args.csv_dir, 'r') as f:
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
    del df_elines_clean
    del df

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
    ###############################################################
    L = Lines()
    consts_K01 = L.consts['K01']
    consts_K01_SII_Ha = L.consts['K01_SII_Ha']
    consts_K01_OI_Ha = L.consts['K01_OI_Ha']
    consts_K03 = L.consts['K03']
    consts_S06 = L.consts['S06']
    consts_K06_SII_Ha = L.consts['K06_SII_Ha']
    consts_K06_OI_Ha = L.consts['K06_OI_Ha']
    if args.no_sigma_clip is False:
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
    y_mod_K06_SII = log_SII_Ha_cen.apply(linval, args=consts_K06_SII_Ha)
    ###############################################################
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [OI]
    ###############################################################
    y_mod_K01_OI = log_OI_Ha_cen.apply(fBPT, args=consts_K01_OI_Ha)
    y_mod_K06_OI = log_OI_Ha_cen.apply(linval, args=consts_K06_OI_Ha)
    ###############################################################
    ###############################################################
    # SELECTIONS
    ###############################################################
    ###############################################################
    sel_NIIHa = ~(log_NII_Ha_cen.apply(np.isnan))
    sel_OIIIHb = ~(log_OIII_Hb_cen.apply(np.isnan))
    sel_SIIHa = ~(log_SII_Ha_cen.apply(np.isnan))
    sel_OIHa = ~(log_OI_Ha_cen.apply(np.isnan))
    sel_EW = ~(EW_Ha_cen.apply(np.isnan))
    sel_AGNLINER_NIIHa_OIIIHb = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01)
    sel_AGN_SIIHa_OIIIHb_K01 = sel_SIIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_SII)
    sel_AGN_OIHa_OIIIHb_K01 = sel_OIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_OI)
    sel_AGN_SIIHa_OIIIHb_K01_K06 = sel_SIIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_SII) & (log_OIII_Hb_cen > y_mod_K06_SII)
    sel_AGN_OIHa_OIIIHb_K01_K06 = sel_OIHa & sel_OIIIHb & (log_OIII_Hb_cen > y_mod_K01_OI) & (log_OIII_Hb_cen > y_mod_K06_OI)
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01
    sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01
    sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 = sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01 & sel_AGN_OIHa_OIIIHb_K01
    sel_AGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug))
    sel_SAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug))
    sel_VSAGN_candidates = (sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01 & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug))
    sel_SF_NIIHa_OIIIHb_K01 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_K01)
    sel_SF_NIIHa_OIIIHb_K03 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_K03)
    sel_SF_NIIHa_OIIIHb_S06 = sel_NIIHa & sel_OIIIHb & (log_OIII_Hb_cen <= y_mod_S06)
    sel_pAGB = sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen <= args.EW_hDIG)
    sel_SF_EW = sel_NIIHa & sel_OIIIHb & sel_EW & (EW_Ha_cen > args.EW_SF*args.bug)
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
    N_GAS = len(elines.loc[~(elines['log_NII_Ha_cen'].apply(np.isnan)) & ~(elines['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
    N_NO_GAS = N_TOT - N_GAS
    ###############################################################
    # [OIII] vs [NII]
    ###############################################################
    # AGN/LINER
    ###############################################################
    m = sel_AGNLINER_NIIHa_OIIIHb
    N_AGN_NII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SAGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    # elines.loc[(m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)), 'TYPE'] = 7
    N_VSAGN_NII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    # SF
    m = sel_SF_NIIHa_OIIIHb_K01
    N_SF_K01 = m.values.astype('int').sum()
    N_SF_K01_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SSF_K01 = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSSF_K01 = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    m = sel_SF_NIIHa_OIIIHb_K03
    N_SF_K03 = m.values.astype('int').sum()
    N_SF_K03_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)), 'TYPE'] = 6
    N_SSF_K03 = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)), 'TYPE'] = 1
    N_VSSF_K03 = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    m = sel_SF_NIIHa_OIIIHb_S06
    N_SF_S06 = m.values.astype('int').sum()
    N_SF_S06_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SSF_S06 = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSSF_S06 = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII]
    ###############################################################
    m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01
    N_AGN_NII_SII_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SAGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_NII_SII_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [OI]
    ###############################################################
    m = sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01
    N_AGN_NII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    N_SAGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    N_VSAGN_NII_OI_Ha_EW = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
    ###############################################################
    # [OIII] vs [NII] + [OIII] vs [SII] + [OIII] vs [OI]
    ###############################################################
    ###############################################################
    m = sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01
    N_AGN_NII_SII_OI_Ha = m.values.astype('int').sum()
    # plus EW(Ha)
    N_AGN = (m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > args.EW_hDIG*args.bug)), 'TYPE'] = 2
    N_SAGN = (m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)).values.astype('int').sum()
    elines.loc[(m & sel_EW & (EW_Ha_cen > args.EW_strong*args.bug)), 'TYPE'] = 3
    N_VSAGN = (m & sel_EW & (EW_Ha_cen > args.EW_verystrong*args.bug)).values.astype('int').sum()
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
    m = (EW_Ha_cen > args.EW_strong) & (sel_AGNLINER_NIIHa_OIIIHb | sel_AGN_SIIHa_OIIIHb_K01 | sel_AGN_OIHa_OIIIHb_K01)
    elines.loc[m, 'AGN_FLAG'] = 4
    m = (EW_Ha_cen > args.EW_strong) & ((sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_SIIHa_OIIIHb_K01) | (sel_AGNLINER_NIIHa_OIIIHb & sel_AGN_OIHa_OIIIHb_K01) | (sel_AGN_SIIHa_OIIIHb_K01 & sel_AGN_OIHa_OIIIHb_K01))
    elines.loc[m, 'AGN_FLAG'] = 3
    m = (elines['TYPE'] == 2) | (elines['TYPE'] == 3)
    elines.loc[m, 'AGN_FLAG'] = 2
    # m = ((elines['SN_broad'] > 8) & ((elines['TYPE'] == 2) | (elines['TYPE'] == 3))) | (elines['broad_by_eye'] == True)
    m = (elines['SN_broad'] > args.min_SN_broad) & (elines['AGN_FLAG'] == 2) | (elines['broad_by_eye'] == True)
    if args.broad_fit_rules is True:
        m = (elines['SN_broad'] > args.min_SN_broad)
    elines.loc[m, 'AGN_FLAG'] = 1
    N_AGN_tI = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 1].count()
    N_AGN_tII = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 2].count()
    N_AGNLINER_N2Ha = elines['AGN_FLAG'].loc[elines['AGN_FLAG'] == 3].count()
    columns_to_csv = [
        'AGN_FLAG', 'SN_broad',
        'RA', 'DEC', 'log_NII_Ha_cen', 'log_NII_Ha_cen_stddev',
        'log_OIII_Hb_cen_mean', 'log_OIII_Hb_cen_stddev',
        'log_SII_Ha_cen_mean', 'log_SII_Ha_cen_stddev',
        'log_OI_Ha_cen', 'e_log_OI_Ha_cen',
        'EW_Ha_cen_mean', 'EW_Ha_cen_stddev',
    ]
    elines.loc[elines['AGN_FLAG'] > 0].to_csv('%s/AGN_CANDIDATES.csv' % args.csv_dir, columns=columns_to_csv)
    # OUTPUT ######################################################################
    print '#AC##################'
    print '#AC# %sN.TOTAL%s = %d' % (color.B, color.E, N_TOT)
    print '#AC# %sN.NO GAS%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS)
    print '#AC##################'
    print '#AC# %sEW cuts%s:' % (color.B, color.E)
    print '#AC# \t%snot-pAGB%s (%sN%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, args.EW_hDIG, args.bug)
    print '#AC# \t%sStrong%s (%sS%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, args.EW_strong, args.bug)
    print '#AC# \t%sVery strong%s (%sVS%s): EW > %d * %.2f A' % (color.B, color.E, color.B, color.E, args.EW_verystrong, args.bug)
    print '#AC# \t%sSF%s: EW > %d A' % (color.B, color.E, args.EW_SF)
    print '#AC##################'
    print '#AC# N.AGNs/LINERs candidates by [NII]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_Ha, color.B, color.E, N_AGN_NII_Ha_EW, color.B, color.E, N_SAGN_NII_Ha_EW, color.B, color.E, N_VSAGN_NII_Ha_EW)
    print '#AC# N.AGNs candidates by [NII]/Ha and [SII]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_Ha, color.B, color.E, N_AGN_NII_SII_Ha_EW, color.B, color.E, N_SAGN_NII_SII_Ha_EW, color.B, color.E, N_VSAGN_NII_SII_Ha_EW)
    print '#AC# N.AGNs candidates by [NII]/Ha and [OI]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_OI_Ha, color.B, color.E, N_AGN_NII_OI_Ha_EW, color.B, color.E, N_SAGN_NII_OI_Ha_EW, color.B, color.E, N_VSAGN_NII_OI_Ha_EW)
    print '#AC# N.AGNs candidates by [NII]/Ha, [SII]/Ha and [OI]/Ha: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (N_AGN_NII_SII_OI_Ha, color.B, color.E, N_AGN, color.B, color.E, N_SAGN, color.B, color.E, N_VSAGN)
    print '#AC# N.AGNs %sAGN/LINER%s: %d - %sType-II%s: %d - %sType-I%s: %d' % (color.B, color.E, N_AGNLINER_N2Ha, color.B, color.E, N_AGN_tII, color.B, color.E, N_AGN_tI)
    print '#AC# N.pAGB: %d (%sabove K01%s: %d - %sabove K03%s: %d - %sabove S06%s: %d)' % (N_pAGB, color.B, color.E, N_pAGB_aboveK01, color.B, color.E, N_pAGB_aboveK03, color.B, color.E, N_pAGB_aboveK03)
    print '#AC# N_SF %sEW%s: %d' % (color.B, color.E, N_SF_EW)
    print '#AC# N.SF %sK01%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K01, color.B, color.E, N_SF_K01_EW, color.B, color.E, N_SSF_K01, color.B, color.E, N_VSSF_K01)
    print '#AC# N.SF %sK03%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_K03, color.B, color.E, N_SF_K03_EW, color.B, color.E, N_SSF_K03, color.B, color.E, N_VSSF_K03)
    print '#AC# N.SF %sS06%s: %d (%sN%s: %d - %sS%s: %d - %sVS%s: %d)' % (color.B, color.E, N_SF_S06, color.B, color.E, N_SF_S06_EW, color.B, color.E, N_SSF_S06, color.B, color.E, N_VSSF_S06)
    print '#AC##################\n'

    elines_wmorph = elines.loc[elines['morph'] >= 0]
    N_TOT_WMORPH = len(elines_wmorph.index)
    N_GAS_WMORPH = len(elines_wmorph.loc[~(elines_wmorph['log_NII_Ha_cen'].apply(np.isnan)) & ~(elines_wmorph['log_OIII_Hb_cen_mean'].apply(np.isnan))].index)
    N_NO_GAS_WMORPH = N_TOT_WMORPH - N_GAS_WMORPH
    print '###################'
    print '## Morph studies ##'
    print '###################\n'
    print '#AC##################'
    print '#AC# %sN.TOTAL WITH MORPHOLOGY%s = %d' % (color.B, color.E, N_TOT_WMORPH)
    print '#AC# %sN.NO GAS WITH MORPHOLOGY%s (without %s[NII]/Ha%s and %s[OIII]/Hb%s) = %d' % (color.B, color.E, color.B, color.E, color.B, color.E, N_NO_GAS_WMORPH)
    print '#AC##################\n'
    ###############################################################################
    # END REPORTS #################################################################
    ###############################################################################

    to_save = {
        'df': elines,
        'sel_NIIHa': sel_NIIHa,
        'sel_OIIIHb': sel_OIIIHb,
        'sel_SIIHa': sel_SIIHa,
        'sel_OIHa': sel_OIHa,
        'sel_EW': sel_EW,
        'sel_AGNLINER_NIIHa_OIIIHb': sel_AGNLINER_NIIHa_OIIIHb,
        'sel_SF_NIIHa_OIIIHb_K01': sel_SF_NIIHa_OIIIHb_K01,
        'sel_SF_NIIHa_OIIIHb_K03': sel_SF_NIIHa_OIIIHb_K03,
        'sel_SF_NIIHa_OIIIHb_S06': sel_SF_NIIHa_OIIIHb_S06,
        'sel_AGN_SIIHa_OIIIHb_K01': sel_AGN_SIIHa_OIIIHb_K01,
        'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01,
        'sel_AGN_OIHa_OIIIHb_K01': sel_AGN_OIHa_OIIIHb_K01,
        'sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_OIHa_K01,
        'sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01': sel_AGN_NIIHa_OIIIHb_K01_SIIHa_K01_OIHa_K01,
        'sel_AGN_candidates': sel_AGN_candidates,
        'sel_SAGN_candidates': sel_SAGN_candidates,
        'sel_VSAGN_candidates': sel_VSAGN_candidates,
        'sel_pAGB': sel_pAGB,
        'sel_SF_EW': sel_SF_EW,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
