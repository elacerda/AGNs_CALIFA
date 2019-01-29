#!/bin/bash
pdftk fig_BPT.pdf fig_histo_CMD_CUBES.pdf fig_histo_CMD_NSA.pdf fig_SFMS.pdf \
    fig_histo_SFMS.pdf fig_SFMS_NC.pdf fig_histo_SFMS_NC.pdf fig_M_C.pdf \
    fig_histo_M_C.pdf fig_M_sSFR.pdf fig_histo_M_sSFR.pdf fig_sSFR_C.pdf \
    fig_histo_sSFR_C.pdf fig_Morph_M.pdf fig_Morph_C.pdf \
    fig_Morph_SigmaMassCen.pdf fig_Morph_vsigma.pdf fig_Morph_Re.pdf \
    cat output allfigs.pdf
