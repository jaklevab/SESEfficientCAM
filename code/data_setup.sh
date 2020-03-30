#!/usr/bin/env bash

# Set up the data directory
mkdir -p ../data/census_data/
cd ../data/census_data/
wget https://www.insee.fr/fr/statistiques/fichier/4176290/Filosofi2015_carreaux_200m_shp.zip
unzip Filosofi2015_carreaux_200m_shp.zip
cd ../../code/

mkdir -p ../data/aerial_data/
# Follow instructions in README.md

mkdir -p ../data/UA_data/
# Follow instructions in README.md

# Set up the results directory
mkdir -p ../results/imagery_out/imag_inter_OUTPUT/
mkdir -p ../results/model_data/logs/
mkdir -p ../results/tmp/
