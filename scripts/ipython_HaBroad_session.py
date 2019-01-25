import numpy as np
import scipy.stats as st

# run agns.py

x = elines

NIIdoubHaNB_r = np.log10((x['NII_6583']+x['NII_6548'])/(x['Ha_narrow'] + x['Ha_broad']))
NIIdoubHaB_r = np.log10((x['NII_6583']+x['NII_6548'])/x['Ha_broad'])
NIIdoubHaN_r = np.log10((x['NII_6583']+x['NII_6548'])/x['Ha_narrow'])
NII6583HaN_r = np.log10(x['NII_6583']/x['Ha_narrow'])
NII6583HaB_r = np.log10(x['NII_6583']/x['Ha_broad'])
NII6583HaNB_r = np.log10(x['NII_6583']/(x['Ha_narrow'] + x['Ha_broad']))

d = {
    'NIIdoubHaNB_r': NIIdoubHaNB_r,
    'NIIdoubHaB_r': NIIdoubHaB_r,
    'NIIdoubHaN_r': NIIdoubHaN_r,
    'NII6583HaN_r': NII6583HaN_r,
    'NII6583HaB_r': NII6583HaB_r,
    'NII6583HaNB_r': NII6583HaNB_r,
}

m = x['log_NII_Ha_cen_mean'].notnull()
 
for k, v in d.iteritems():
    tmp_m = m & v.notnull()
    s = st.spearmanr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    p = st.pearsonr(v[tmp_m ], x['log_NII_Ha_cen_mean'][tmp_m])
    print '%s - s:%.3f p:%3f' % (k, s[0], p[0])

m_nAGNs = x['AGN_FLAG'] == 0

for k, v in d.iteritems():
    tmp_m = m & v.notnull() & m_nAGNs
    snAGN = st.spearmanr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    pnAGN = st.pearsonr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    print 'non-AGNs - %s - s:%.3f p:%3f' % (k, snAGN[0], pnAGN[0])

m_AGNs = x['AGN_FLAG'] > 0

for k, v in d.iteritems():
    tmp_m = m & v.notnull() & m_AGNs
    sAGN = st.spearmanr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    pAGN = st.pearsonr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    print 'AGNs - %s - s:%.3f p:%3f' % (k, sAGN[0], pAGN[0])

m_AGNtI = x['AGN_FLAG'] == 1

for k, v in d.iteritems():
    tmp_m = m & v.notnull() & m_AGNtI
    sAGNtI = st.spearmanr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    pAGNtI = st.pearsonr(v[tmp_m], x['log_NII_Ha_cen_mean'][tmp_m])
    print 'tI AGNs - %s - s:%.3f p:%3f' % (k, sAGNtI[0], pAGNtI[0])
