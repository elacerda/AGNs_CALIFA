#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage: $0 EWAGN BUG"
	exit 1
fi

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
echo "$0: running python3 agns_tables.py -O ${OUTPUTTABLES} --csv_dir=${CSVPATH}"
python3 agns_tables.py -O ${OUTPUTTABLES} --csv_dir=${CSVPATH}
echo -e "\n"
echo "##########################"
echo "## Generating selection ##"
echo "##########################"
#BUG=0.8
EWAGN=$1
BUG=$2
RUNTAG=EWAGN${EWAGN}_BUG${BUG}
OUTPUTAGNSFILE=AGN_CANDIDATES_${RUNTAG}.csv
OUTPUTFILE="${DATAPATH}/elines_${RUNTAG}.pkl"
echo "$0: running python3 agns_selection.py -I ${OUTPUTTABLES} -O ${OUTPUTFILE} --bug=${BUG} --EW_AGN=${EWAGN} --csv_dir=${CSVPATH} --output_agn_candidates=${OUTPUTAGNSFILE}"
python3 agns_selection.py -I ${OUTPUTTABLES} -O ${OUTPUTFILE} --bug=${BUG} --EW_AGN=${EWAGN} --csv_dir=${CSVPATH} --output_agn_candidates=${OUTPUTAGNSFILE}
#--no_sigma_clip
echo "#########"
echo "## END ##"
echo "#########"
echo "$0: To generate plots run python agns_plots.py -I ${OUTPUTFILE} -vv"
