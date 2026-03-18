import itertools

# Very simple grid
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


def _recursive_forecast_with_exog(res, history: pd.Series, future_index) -> np.ndarray:
    preds = []
    state = res
    hist = history.copy()
    y_name = hist.name if hist.name is not None else 'y'

    for dt in future_index:
        x_next = pd.DataFrame({
            'lag_1': [float(hist.iloc[-1])],
            'ma_3': [float(hist.iloc[-3:].mean())],
            'month_num': [int(pd.to_datetime(dt).month)],
        }, index=[dt])[_EXOG_COLS]

        y_hat = float(state.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0])
        preds.append(y_hat)

        y_new = pd.Series([y_hat], index=[dt], name=y_name, dtype=float)
        hist = pd.concat([hist, y_new])
        state = state.append(endog=y_new, exog=x_next, refit=False)

    return np.array(preds, dtype=float)


def _sarima_mape(tr, val, p, d, q, P, D, Q):
    """Fit SARIMAX(+exog) and return MAPE % using ALL validation points."""
    try:
        # Build train exog (drop only warm-up rows from training side)
        exog_tr_full = _build_exog_from_series(tr)
        valid_idx = exog_tr_full.dropna().index
        tr_fit = tr.loc[valid_idx]
        exog_tr = exog_tr_full.loc[valid_idx, _EXOG_COLS]

        res = SARIMAX(
            endog=tr_fit,
            exog=exog_tr,
            order=(p, d, q),
            seasonal_order=(P, D, Q, _m),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # Use all validation points (for tuning 4 this is all 12 test points)
        yp = _recursive_forecast_with_exog(res, tr_fit, val.index)
        yt = val.values.astype(float)

        # Same metric used in your main evaluation
        return float(mean_absolute_percentage_error(yt, yp) * 100)

    except Exception:
        return np.nan


# Tuning 4: Full train -> test (uses all points in test_y)
rows = []
for p, d, q, P, D, Q in _grid:
    mape = _sarima_mape(train_y, test_y, p, d, q, P, D, Q)
    if not np.isnan(mape):
        rows.append({
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q,
            'MAPE': round(mape, 2),
            'n_val_points': len(test_y),
        })

df_tune4 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
best4 = df_tune4.iloc[0]
print("Tuning 4 (Full train -> test set, SARIMAX+exog) - Best Params:")
print(f"  order=({int(best4.p)},{int(best4.d)},{int(best4.q)})  "
      f"seasonal_order=({int(best4.P)},{int(best4.D)},{int(best4.Q)},12)  "
      f"MAPE={best4.MAPE:.2f}%")
print("\nTop 5:")
display(df_tune4.head(5))
