#!/bin/bash
FIGS_DIR=$1
if [ ! -d "$FIGS_DIR" ]
then
    FIGS_DIR=${HOME}/dev/astro/AGNs_CALIFA/figs
fi
PUBDIR=${HOME}/public_html/AGNs_CALIFA/
ALLFIGS=$FIGS_DIR/allfigs.pdf
HISTOFIGS=$FIGS_DIR/allfigs_histo.pdf
MORPHFIGS=$FIGS_DIR/allfigs_Morph.pdf
FIGS=()
FIGS_HISTO=()
FIGS_MORPH=()

# Remove old ALLFIGS file
if [ -f "$ALLFIGS" ]
then
  echo "$0: removing $ALLFIGS"
  rm $ALLFIGS
fi
if [ -f "$HISTOFIGS" ]
then
  echo "$0: removing $HISTOFIGS"
  rm $HISTOFIGS
fi
if [ -f "$MORPHFIGS" ]
then
  echo "$0: removing $MORPHFIGS"
  rm $MORPHFIGS
fi

# Search for all pdf files
for f in $FIGS_DIR/*.pdf
do
  FIGS+=( $f )
done
for f in $FIGS_DIR/*histo*.pdf
do
  FIGS_HISTO+=( $f )
done
for f in $FIGS_DIR/*Morph*.pdf
do
  FIGS_MORPH+=( $f )
done

# Generate a pdf with all figs
if [ -z "$( which pdftk )" ]
then
  echo "$0: using qpdf instead pdftk"
  qpdf --empty --pages ${FIGS[*]} -- $ALLFIGS
  qpdf --empty --pages ${FIGS_HISTO[*]} -- $HISTOFIGS
  qpdf --empty --pages ${FIGS_MORPH[*]} -- $MORPHFIGS
else
  pdftk ${FIGS[*]} cat output $ALLFIGS
  pdftk ${FIGS_HISTO[*]} cat output $HISTOFIGS
  pdftk ${FIGS_MORPH[*]} cat output $MORPHFIGS
fi

# Publish the images on localhost
cp -r ${FIGS_DIR} ${PUBDIR}
