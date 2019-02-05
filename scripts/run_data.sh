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
./agns_tables.py -O ${DATADIR}/dataframes.pkl --csv_dir=csv
echo -e "\n"
echo "##########################"
echo "## Generating selection ##"
echo "##########################"
./agns_selection.py -I ${DATADIR}/dataframes.pkl -O ${DATADIR}/elines.pkl
echo "#########"
echo "## END ##"
echo "#########"
echo "$0: To generate plots run ./agns_plots.py -I ${DATADIR}/elines.pkl "
