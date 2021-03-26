#!/bin/bash
source /home/user_data/anaconda3/etc/profile.d/conda.sh
conda activate py_maoyl
command -v python
cd ~/GeoQA
python search.py
conda deactivate

conda activate py_maoyl_qa
command -v python
cd ~/nlp_scqa_geo/code
python geo_qa_app.py
conda deactivate
