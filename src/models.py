import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import CLIP_HIGH, CLIP_LOW, FEATURE_SETS, FEATURES, SPLITS, XGB_PARAMS


def make_splits(df):
    """Chronological splits by trading day. Clipping limits come from the training days only."""
    days = np.sort(df['date_id'].unique())
    splits = {}
    for name, frac in SPLITS.items():
        cut = days[int(frac * len(days))]
        train = df[df['date_id'] < cut]
        test = df[df['date_id'] >= cut]

        lo, hi = train[FEATURES].quantile(CLIP_LOW), train[FEATURES].quantile(CLIP_HIGH)
        t_lo, t_hi = train['target'].quantile([CLIP_LOW, CLIP_HIGH])

        splits[name] = {
            'X_train': train[FEATURES].clip(lo, hi, axis=1),
            'y_train': train['target'],
            'y_fit': train['target'].clip(t_lo, t_hi),
            'X_test': test[FEATURES].clip(lo, hi, axis=1),
            'y_test': test['target'],
            'stock_test': test['stock_id'],
            'train_days': (days[0], cut - 1),
            'test_days': (cut, days[-1]),
        }
    return splits


def build_models():
    return {
        'Linear': LinearRegression(),
        'Polynomial (deg 2)': make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        ),
        'XGBoost': xgb.XGBRegressor(**XGB_PARAMS),
    }


def direction_accuracy(y_true, y_pred):
    return (np.sign(y_true) == np.sign(y_pred)).mean() * 100


def improvement(base_mae, mae):
    return (base_mae - mae) / base_mae * 100


def run_experiments(splits, verbose=True):
    """Naive baseline plus 3 models x 3 feature sets for every split."""
    results, predictions = [], {}
    for split_name, s in splits.items():
        base_pred = np.full(len(s['y_test']), s['y_train'].mean())
        base_mae = mean_absolute_error(s['y_test'], base_pred)
        results.append({
            'Split': split_name, 'Features': '-', 'Model': 'Naive baseline',
            'MAE': base_mae, 'Direction %': direction_accuracy(s['y_test'], base_pred), 'Improvement %': 0.0,
        })

        for set_name, cols in FEATURE_SETS.items():
            for model_name, model in build_models().items():
                model.fit(s['X_train'][cols], s['y_fit'])
                pred = model.predict(s['X_test'][cols])
                predictions[(split_name, set_name, model_name)] = pred
                mae = mean_absolute_error(s['y_test'], pred)
                results.append({
                    'Split': split_name, 'Features': set_name, 'Model': model_name,
                    'MAE': mae, 'Direction %': direction_accuracy(s['y_test'], pred),
                    'Improvement %': improvement(base_mae, mae),
                })
                if verbose:
                    print(f'{split_name} | {set_name:<14} | {model_name:<18} | MAE {mae:.4f}')
    return pd.DataFrame(results), predictions


def per_stock_results(splits, predictions, tiers, set_name='C: + Market', model_name='XGBoost'):
    """Scores one model on each stock's test rows against the baseline on the same rows."""
    rows = []
    for split_name, s in splits.items():
        pred = pd.Series(predictions[(split_name, set_name, model_name)], index=s['y_test'].index)
        train_mean = s['y_train'].mean()
        for sid, tier in tiers.items():
            idx = s['stock_test'] == sid
            y = s['y_test'][idx]
            mae = mean_absolute_error(y, pred[idx])
            base = mean_absolute_error(y, np.full(len(y), train_mean))
            rows.append({
                'Split': split_name, 'Stock': sid, 'Tier': tier, 'Test rows': len(y),
                'XGBoost MAE': mae, 'Baseline MAE': base, 'Improvement %': improvement(base, mae),
                'Direction %': direction_accuracy(y, pred[idx]),
            })
    return pd.DataFrame(rows)
