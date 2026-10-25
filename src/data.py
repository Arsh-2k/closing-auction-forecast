import numpy as np
import pandas as pd

from .config import AUCTION, FEATURES, PREPROCESSED_PATH, RAW_PATH, SAVE_COLUMNS


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def load_preprocessed(path=PREPROCESSED_PATH):
    return pd.read_csv(path, float_precision='round_trip')


def select_stocks(raw):
    """3 most liquid, the median and the least liquid stock, ranked by average matched_size (USD)."""
    liquidity = raw.groupby('stock_id')['matched_size'].mean().sort_values(ascending=False)
    ranked = liquidity.index.tolist()
    tiers = {sid: 'High' for sid in ranked[:3]}
    tiers[ranked[len(ranked) // 2]] = 'Mid'
    tiers[ranked[-1]] = 'Low'
    return tiers, liquidity


def add_features(raw, stocks):
    # market averages need every stock, so these are computed before filtering
    df = raw.sort_values(['stock_id', 'date_id', 'seconds_in_bucket']).reset_index(drop=True)
    df['imbalance_ratio'] = df['imbalance_size'] * df['imbalance_buy_sell_flag'] / (df['matched_size'] + 1)
    wap_60s_ago = df.groupby(['stock_id', 'date_id'])['wap'].shift(6)
    df['return_60s'] = (df['wap'] / wap_60s_ago - 1) * 1e4

    market = df.groupby(['date_id', 'seconds_in_bucket'])[['imbalance_ratio', 'return_60s']].transform('mean')
    df['rel_imbalance'] = df['imbalance_ratio'] - market['imbalance_ratio']
    df['rel_return_60s'] = df['return_60s'] - market['return_60s']

    d = df[df['stock_id'].isin(stocks)].copy()
    d['liquidity_imbalance'] = (d['bid_size'] - d['ask_size']) / (d['bid_size'] + d['ask_size'] + 1)
    d['spread'] = (d['ask_price'] - d['bid_price']) / d['wap'] * 1e4
    d['ref_price_dev'] = (d['wap'] - d['reference_price']) / d['reference_price'] * 1e4
    d['near_price_dev'] = (d['near_price'] - d['reference_price']) / d['reference_price'] * 1e4
    d['far_price_dev'] = (d['far_price'] - d['reference_price']) / d['reference_price'] * 1e4
    return d


def clean(df):
    df = df.dropna(subset=['target']).copy()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df[AUCTION] = df[AUCTION].fillna(0)
    df[['return_60s', 'rel_return_60s']] = df[['return_60s', 'rel_return_60s']].fillna(0)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def build_dataset(raw, tiers):
    df = clean(add_features(raw, list(tiers)))
    df['liquidity_tier'] = df['stock_id'].map(tiers)
    return df


def save_preprocessed(df, path=PREPROCESSED_PATH):
    df[SAVE_COLUMNS].to_csv(path, index=False)
