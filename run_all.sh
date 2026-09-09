#!/usr/bin/env bash
set -euo pipefail

python prepare_data.py
python train_baseline.py
python train_model.py
streamlit run app.py
