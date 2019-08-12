#!/bin/bash
# Instalador del Entorno
clear

clear
conda env create -f envs/base_enviroment.yml
conda deactivate
conda install ipykernel --name music_clasification
conda env update -f envs/local_enviroment.yml

DATASET=data/genres.tar.gz
if test -f "$DATASET"; then
    echo "$DATASET existe"
    echo "Descomprimiendo"
    tar -xvzf data/genres.tar.gz -C data/
else
    echo "Descargando GTZAN dataset"
    cd data && { curl -O http://opihi.cs.uvic.ca/sound/genres.tar.gz ; cd ..; }
    echo "Descomprimiendo"
    tar -xvzf data/genres.tar.gz -C data/
fi