#!/bin/bash
DATADIR=data
if [ ! -d "${DATADIR}" ]
then
    mkdir -p "${DATADIR}"
else
    echo "$0: directory ${DATADIR} already exists"
fi
echo "#######################"
echo "## Generating tables ##"
echo "#######################"
OUTPUT_TABLES="${DATADIR}/tables.pkl"
./agns_tables.py -O ${OUTPUT_TABLES} --csv_dir=csv
echo -e "\n"
echo "##########################"
echo "## Generating selection ##"
echo "##########################"
#BUG=0.8
BUG=1
#EW_AGN=1.875
EW_AGN=3
OUTPUT_FILE="${DATADIR}/elines_EWAGN${EW_AGN}_BUG${BUG}.pkl"
./agns_selection.py -I ${OUTPUT_TABLES} -O ${OUTPUT_FILE} --bug=${BUG} --EW_AGN=${EW_AGN}
#--no_sigma_clip
echo "#########"
echo "## END ##"
echo "#########"
echo "$0: To generate plots run ./agns_plots.py -I ${OUTPUT_FILE} -vv"
