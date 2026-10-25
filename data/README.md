# Data

The data files are not stored in this repository.

1. Open the Kaggle competition [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data) and accept the competition rules.
2. Download `train.csv` and place it in this folder as `data/train.csv`.

Running the notebook or `run_pipeline.py` creates `data/optiver_preprocessed.csv` (132,275 rows x 14 columns): the five selected stocks, their liquidity tier, the ten features and the target. Both CSV files are ignored by git.
