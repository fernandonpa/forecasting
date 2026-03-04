%%capture
!pip install --proxy=192.168.2.10:8080 --upgrade snowflake-connector-python[pandas]
!pip install --proxy=192.168.2.10:8080 --upgrade keyring seaborn
!pip install --proxy=192.168.2.10:8080 prophet
!pip install --proxy=192.168.2.10:8080 hyperopt    # ← ADD THIS LINE




import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import snowflake.connector
import datetime
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# ← ADD THESE 3 LINES
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


# ── Walk-forward CV splits (no data leakage) ─────────────────────────────
def ts_cv_splits(series, n_splits=3, val_size=2, min_train=10):
    n = len(series)
    for fold in range(n_splits):
        val_end   = n - (n_splits - 1 - fold) * val_size
        val_start = val_end - val_size
        if val_start < min_train:
            continue
        yield list(range(val_start)), list(range(val_start, val_end))


# ── Hyperopt search space ─────────────────────────────────────────────────
prophet_space = {
    'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale',
                                              np.log(0.001), np.log(10)),
    'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale',
                                              np.log(0.01), np.log(5)),
    'changepoint_range'      : hp.uniform('changepoint_range', 0.70, 0.95),
    'n_changepoints'         : hp.quniform('n_changepoints', 5, 30, 1),
}


# ── CV objective function ─────────────────────────────────────────────────
def prophet_cv_objective(params):
    metric = 'Beginning Outstanding Loans'   # ← change to your target metric
    series = train_df[metric].dropna()
    fold_mapes = []

    for tr_idx, val_idx in ts_cv_splits(series, n_splits=3, val_size=2, min_train=10):
        tr  = series.iloc[tr_idx]
        val = series.iloc[val_idx]

        df_p = pd.DataFrame({'ds': pd.to_datetime(tr.index), 'y': tr.values})
        df_p['floor'] = 0
        df_p['cap']   = df_p['y'].max() * 1.2

        df_val = pd.DataFrame({
            'ds'   : pd.to_datetime(val.index),
            'floor': 0,
            'cap'  : df_p['y'].max() * 1.2
        })

        try:
            m = Prophet(
                growth='logistic',
                changepoint_prior_scale = params['changepoint_prior_scale'],
                seasonality_prior_scale = params['seasonality_prior_scale'],
                changepoint_range       = params['changepoint_range'],
                n_changepoints          = int(params['n_changepoints']),
                seasonality_mode        = 'additive',
                weekly_seasonality      = False,
                daily_seasonality       = False,
                yearly_seasonality      = False,
            )
            m.fit(df_p)
            fc   = m.predict(df_val)['yhat'].values
            fc   = np.maximum(fc, 0)
            mape = mean_absolute_percentage_error(val.values, fc) * 100
            fold_mapes.append(mape)
        except Exception:
            fold_mapes.append(200)   # heavy penalty for crashes

    return {'loss': np.mean(fold_mapes), 'status': STATUS_OK}


# ── Run Hyperopt (50 trials) ──────────────────────────────────────────────
print("Running Hyperopt... please wait")
prophet_trials = Trials()

best_raw = fmin(
    fn        = prophet_cv_objective,
    space     = prophet_space,
    algo      = tpe.suggest,
    max_evals = 50,
    trials    = prophet_trials,
    rstate    = np.random.default_rng(42),
    verbose   = False,
)

best_prophet_params = space_eval(prophet_space, best_raw)
best_prophet_params['n_changepoints'] = int(best_prophet_params['n_changepoints'])

print(f"\nBest CV MAPE : {min(prophet_trials.losses()):.2f}%")
print(f"Senior baseline : 6.54%")
print(f"Statistical target : 3.21%")
print("\nBest params:")
for k, v in best_prophet_params.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")



model = Prophet(growth='logistic',
                changepoint_prior_scale = best_prophet_params['changepoint_prior_scale'],
                seasonality_prior_scale = best_prophet_params['seasonality_prior_scale'],
                changepoint_range       = best_prophet_params['changepoint_range'],
                n_changepoints          = best_prophet_params['n_changepoints'],
                seasonality_mode        = 'additive',
                weekly_seasonality      = False,
                daily_seasonality       = False,
                yearly_seasonality      = False,
)



