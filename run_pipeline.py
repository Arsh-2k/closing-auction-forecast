"""Runs the full Milestone 2 pipeline without Jupyter.

    python run_pipeline.py

Reads data/train.csv, writes data/optiver_preprocessed.csv,
results/experiments.csv and results/per_stock.csv.
"""
from src import config, data, models


def main():
    print('Loading', config.RAW_PATH)
    raw = data.load_raw()
    tiers, liquidity = data.select_stocks(raw)
    for sid, tier in tiers.items():
        print(f'stock {sid:>3}  {tier:<4}  avg matched size = {liquidity[sid]:>15,.0f}')

    df = data.build_dataset(raw, tiers)
    data.save_preprocessed(df)
    print(f'Saved {config.PREPROCESSED_PATH.name}: {len(df):,} rows')

    splits = models.make_splits(df)
    results, predictions = models.run_experiments(splits)
    per_stock = models.per_stock_results(splits, predictions, tiers)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.RESULTS_DIR / 'experiments.csv', index=False)
    per_stock.to_csv(config.RESULTS_DIR / 'per_stock.csv', index=False)

    print('\nAll ten features (C: + Market):')
    summary = results[results['Features'].isin(['-', 'C: + Market'])]
    print(summary[['Split', 'Model', 'MAE', 'Direction %', 'Improvement %']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
