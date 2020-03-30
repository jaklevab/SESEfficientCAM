#!/usr/bin/env bash
set -ex

# Execute results
python3 generate_fr_ua_aerial_data.py
python3 efficientnet_training.py
python3 gradcaming_urban_areas.py
