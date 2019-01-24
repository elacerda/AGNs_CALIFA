#!/home/lacerda/anaconda2/bin/python
import numpy as np
import pandas as pd
import os.path as p

elines = pd.read_csv('csv/get_proc_elines_CALIFA.clean.pandas.csv', comment='#', usecols=[0], header='infer', index_col=0)
col = ['Ha_narrow', 'NII_6583', 'NII_6548', 'Ha_broad']
df_broadlines = pd.DataFrame(index=elines.index, columns=col)
del elines
for i in df_broadlines.index:
    broad_f = 'csv/broad/output_Ha_broad_cen.%s.out' % i
    if p.isfile(broad_f):
        f_df = pd.read_csv(broad_f, skiprows=1, usecols=[3], sep=' ', header=None, names=['flux'])
        df_broadlines.loc[i] = f_df['flux'].values
        del f_df
df_broadlines.to_csv('csv/NII_Ha_fit.csv', columns=col)
