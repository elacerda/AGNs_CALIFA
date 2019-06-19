#!/bin/bash

if [ $# -lt 3 ]
then
	echo "Usage: $0 EWAGN BUG IMGSUFFIX"
	exit 1
fi
WORKPATH=${HOME}/dev/astro/AGNs_CALIFA
FIGSPATH=${WORKPATH}/figs
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
echo -e "\n"
echo "######################"
echo "## Generating plots ##"
echo "######################"
IMGSUFFIX=$3
OUTPUTFIGSDIR=${FIGSPATH}/${RUNTAG}_${IMGSUFFIX}
if [ ! -d "${OUTPUTFIGSDIR}" ]
then
    mkdir "${OUTPUTFIGSDIR}" --
fi
PLOTSARGS="-I ${OUTPUTFILE} --figs_dir=${OUTPUTFIGSDIR} -vv --img_suffix=${IMGSUFFIX}"
echo "$0: running python3 ${WORKPATH}/agns_plots.py ${PLOTSARGS}"
python3 ${WORKPATH}/agns_plots.py ${PLOTSARGS}
echo "#########"
echo "## END ##"
echo "#########"
