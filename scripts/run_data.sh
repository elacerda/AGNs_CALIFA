#!/bin/bash
WORKPATH=${HOME}/dev/astro/AGNs_CALIFA
LOGSPATH=${WORKPATH}/logs
DATAPATH=${WORKPATH}/data
CSVPATH=${WORKPATH}/csv
if [ ! -d "${DATAPATH}" ]
then
    echo "$0: creating directory ${DATAPATH}"
    mkdir -p "${DATAPATH}"
fi
echo "#######################"
echo "## Generating tables ##"
echo "#######################"
OUTPUTTABLES="${DATAPATH}/tables.pkl"
echo "$0: running ./agns_tables.py -O ${OUTPUTTABLES} --csv_dir=${CSVPATH}"
./agns_tables.py -O ${OUTPUTTABLES} --csv_dir=${CSVPATH}
echo -e "\n"
echo "##########################"
echo "## Generating selection ##"
echo "##########################"
OUTPUTFILE=${DATAPATH}/elines_${RUNTAG}.pkl
BUG=1
BUG=0.8
EWAGN=3
RUNTAG=EWAGN${EWAGN}_BUG${BUG}
OUTPUT_FILE="${DATAPATH}/elines_${RUNTAG}.pkl"
echo "$0: running ./agns_selection.py -I ${OUTPUTTABLES} -O ${OUTPUTFILE} --bug=${BUG} --EW_AGN=${EWAGN} --csv_dir=${CSVPATH} --output_agn_candidates=${CSVPATH}/AGN_CANDIDATES_${RUNTAG}.csv"
./agns_selection.py -I ${OUTPUTTABLES} -O ${OUTPUTFILE} --bug=${BUG} --EW_AGN=${EWAGN} --csv_dir=${CSVPATH} --output_agn_candidates=${CSVPATH}/AGN_CANDIDATES_${RUNTAG}.csv
#--no_sigma_clip
echo "#########"
echo "## END ##"
echo "#########"
echo "$0: To generate plots run ./agns_plots.py -I ${OUTPUT_FILE} -vv"
