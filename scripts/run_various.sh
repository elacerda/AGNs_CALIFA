#!/bin/bash
BUG=(0.8 1)
EWAGN=(3 6)
#EWAGN=(1.5)
WORKPATH=${HOME}/dev/astro/AGNs_CALIFA
FIGSPATH=${WORKPATH}/figs
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
for b in ${BUG[*]}
do
	for A in ${EWAGN[*]}
	do
		RUNTAG=EWAGN${A}_BUG${b}
		OUTPUTFILE=${DATAPATH}/elines_${RUNTAG}.pkl
		OUTPUTFIGSDIR=${FIGSPATH}/${RUNTAG}
		LOGSEL=${LOGSPATH}/agns_selection_${RUNTAG}.log
		LOGPLOT=${LOGSPATH}/agns_plots_${RUNTAG}.log
		OUTPUTAGNSFILE=AGN_CANDIDATES_${RUNTAG}.csv
		
		if [ ! -d "${OUTPUTFIGSDIR}" ]
		then
			mkdir "${OUTPUTFIGSDIR}" --
		fi

		SELARGS="-I ${DATAPATH}/tables.pkl --csv_dir=${CSVPATH} --EW_AGN=$A --bug=$b -O ${OUTPUTFILE} --output_agn_candidates=${OUTPUTAGNSFILE}"
		PLOTSARGS="-I ${OUTPUTFILE} --figs_dir=${OUTPUTFIGSDIR} -vv"

		echo "$0: running python3 ${WORKPATH}/agns_selection.py ${SELARGS} &> $LOGSEL"
		python3 ${WORKPATH}/agns_selection.py ${SELARGS} &> $LOGSEL
		echo "$0: running python3 ${WORKPATH}/agns_plots.py ${PLOTSARGS} &> $LOGPLOT"
		python3 ${WORKPATH}/agns_plots.py ${PLOTSARGS} &> $LOGPLOT
	done
done
