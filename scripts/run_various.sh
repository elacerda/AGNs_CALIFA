#!/bin/bash
BUG=(0.8 1)
EWAGN=(3 6 10 14)
FIGSPATH=${HOME}/dev/astro/AGNs_CALIFA/figs
LOGSPATH=${HOME}/dev/astro/AGNs_CALIFA/logs
DATAPATH=${HOME}/dev/astro/AGNs_CALIFA/data
for b in ${BUG[*]}
do
	for A in ${EWAGN[*]}
	do
		RUNTAG=EWAGN${A}_BUG${b}
		OUTPUTFILE=${DATAPATH}/elines_${RUNTAG}.pkl
		OUTPUTFIGSDIR=${FIGSPATH}/${RUNTAG}
		LOGSEL=${LOGSPATH}/agns_selection_${RUNTAG}.log
		LOGPLOT=${LOGSPATH}/agns_plots_${RUNTAG}.log
		
		if [ ! -d "${OUTPUTFIGSDIR}" ]
		then
			mkdir "${OUTPUTFIGSDIR}" --
		fi

		./agns_selection.py -I ${DATAPATH}/dataframes.pkl --EW_AGN=$A --bug=$b -O ${OUTPUTFILE} &> $LOGSEL
		./agns_plots.py -I ${OUTPUTFILE} --figs_dir=${OUTPUTFIGSDIR} &> $LOGPLOT
	done
done
