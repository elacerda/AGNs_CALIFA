#!/bin/bash
FIGS_DIR=${HOME}/dev/astro/AGNs_CALIFA/figs
PUBDIR=${HOME}/public_html/AGNs_CALIFA/
ALLFIGS=$FIGS_DIR/allfigs.pdf
FIGS=()

# Remove old ALLFIGS file
if [ -f "$ALLFIGS" ]
then
  echo "$0: removing $ALLFIGS"
  rm $ALLFIGS
fi

# Search for all pdf files
for f in $FIGS_DIR/*.pdf
do
  FIGS+=( $f )
done

# Generate a pdf with all figs
if [ -z "$( which pdftk )" ]
then
  echo "$0: using qpdf instead pdftk"
  qpdf --empty --pages ${FIGS[*]} -- $ALLFIGS
else
  pdftk ${FIGS[*]} cat output $ALLFIGS
fi

# Publish the images on localhost
cp -r ${FIGS_DIR} ${PUBDIR}
