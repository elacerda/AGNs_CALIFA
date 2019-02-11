#!/bin/bash
FIGS_DIR=${HOME}/dev/astro/AGNs_CALIFA/figs
ALLFIGS=$FIGS_DIR/allfigs.pdf
FIGS=()
if [ -f "$ALLFIGS" ]
then
  echo "$0: removing $ALLFIGS"
  rm $ALLFIGS
fi

for f in $FIGS_DIR/*.pdf; do
  FIGS+=( $f )
done

if [ -z "$( which pdftk )" ]
then
  echo "$0: using qpdf instead pdftk"
  qpdf --empty --pages ${FIGS[*]} -- $ALLFIGS
else
  pdftk $FIGS cat output $ALLFIGS
fi
