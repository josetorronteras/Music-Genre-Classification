#!/bin/bash
# Instalador del Entorno
clear

conda env create -f enviroment.yml

GTZAN=data/genres.tar.gz
if test -f "$GTZAN"; then
    echo "$GTZAN existe"
    echo "Descomprimiendo"
    tar -xvzf data/genres.tar.gz -C data/
else
    echo "Descargando GTZAN dataset"
    cd data && { curl -O http://opihi.cs.uvic.ca/sound/genres.tar.gz ; cd ..; }
    echo "Descomprimiendo"
    tar -xvzf data/genres.tar.gz -C data/
fi