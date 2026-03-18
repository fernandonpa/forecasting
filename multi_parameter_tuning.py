import itertools

# ── Shared grid + helper (SARIMAX + new exogenous features) ───────────────
_grid = list(itertools.product(
    [0, 1, 2],  # p
    [0, 1],     # d
    [0, 1, 2],  # q
    [0, 1],     # P
    [0, 1],     # D
    [0, 1, 2],  # Q
))
_m = 12
_EXOG_COLS = ['lag_1', 'ma_3', 'month_num']


def _build_exog_from_series(y: pd.Series) -> pd.DataFrame:
    idx = pd.to_datetime(y.index)
    exog = pd.DataFrame(index=y.index)
    exog['lag_1'] = y.shift(1)
    exog['ma_3'] = y.shift(1).rolling(3).mean()
    exog['month_num'] = idx.month.astype(int)
    return exog


def _sarima_mape(tr, val, p, d, q, P, D, Q):
    """Fit SARIMAX(+exog) and return sklearn MAPE (%) on full validation horizon."""
    try:
        # Build train exog and drop only training warm-up rows from lag/MA features
        exog_tr_full = _build_exog_from_series(tr)
        valid_idx = exog_tr_full.dropna().index
        if len(valid_idx) < 8:
            return np.nan

        tr_fit = tr.loc[valid_idx]
        exog_tr = exog_tr_full.loc[valid_idx, _EXOG_COLS]

        state = SARIMAX(
            endog=tr_fit,
            exog=exog_tr,
            order=(p, d, q),
            seasonal_order=(P, D, Q, _m),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # Recursive prediction on validation horizon (uses all validation dates)
        hist = tr_fit.copy()
        endog_name = hist.name if hist.name is not None else 'y'
        preds = []

        for dt in val.index:
            x_next = pd.DataFrame({
                'lag_1': [float(hist.iloc[-1])],
                'ma_3': [float(hist.iloc[-3:].mean())],
                'month_num': [int(pd.to_datetime(dt).month)],
            }, index=[dt])[_EXOG_COLS]

            y_hat = float(state.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0])
            preds.append(y_hat)

            y_new = pd.Series([y_hat], index=[dt], name=endog_name, dtype=float)
            hist = pd.concat([hist, y_new])
            state = state.append(endog=y_new, exog=x_next, refit=False)

        yt = val.values.astype(float)
        yp = np.array(preds, dtype=float)

        # Ensure no validation points are dropped
        if len(yt) != len(yp):
            return np.nan

        # Exact same metric family as model-evaluation cells
        return float(mean_absolute_percentage_error(yt, yp) * 100)

    except Exception:
        return np.nan


# ── Tuning 4: Full training set → test set (real out-of-sample) ──────────
# train on all of train_y, validate on full test_y horizon
print(f"Validation points used in Tuning 4: {len(test_y)}")
if len(test_y) != 12:
    print("[WARN] Test horizon is not 12 months in current filtered dataset.")

rows = []
for p, d, q, P, D, Q in _grid:
    mape = _sarima_mape(train_y, test_y, p, d, q, P, D, Q)
    if not np.isnan(mape):
        rows.append({
            'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q,
            'MAPE': round(mape, 2),
            'n_val_points': len(test_y),
        })

df_tune4 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
best4 = df_tune4.iloc[0]
print("Tuning 4 (Full train → test set, SARIMAX+exog) — Best Params:")
print(f"  order=({int(best4.p)},{int(best4.d)},{int(best4.q)})  "
      f"seasonal_order=({int(best4.P)},{int(best4.D)},{int(best4.Q)},12)  "
      f"MAPE={best4.MAPE:.2f}%")
print("\nTop 5:")
display(df_tune4.head(5))
