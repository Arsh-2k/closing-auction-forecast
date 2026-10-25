from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_PATH = DATA_DIR / 'train.csv'
PREPROCESSED_PATH = DATA_DIR / 'optiver_preprocessed.csv'
FIGURES_DIR = ROOT / 'figures'
RESULTS_DIR = ROOT / 'results'

ORDER_BOOK = ['imbalance_ratio', 'liquidity_imbalance', 'spread', 'ref_price_dev', 'return_60s']
AUCTION = ['near_price_dev', 'far_price_dev']
MARKET = ['rel_imbalance', 'rel_return_60s', 'seconds_in_bucket']
FEATURES = ORDER_BOOK + AUCTION + MARKET

FEATURE_SETS = {
    'A: Order book': ORDER_BOOK,
    'B: + Auction': ORDER_BOOK + AUCTION,
    'C: + Market': FEATURES,
}

SAVE_COLUMNS = ['stock_id', 'liquidity_tier', 'date_id'] + FEATURES + ['target']

SPLITS = {'70/30': 0.7, '80/20': 0.8}
CLIP_LOW, CLIP_HIGH = 0.01, 0.99

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:absoluteerror',
    'tree_method': 'hist',
    'random_state': 42,
}
