import itertools

# ── Shared grid + helper (used by all 4 tuning cells) ────────────────────
_grid = list(itertools.product(
    [0, 1, 2],  # p
    [0, 1],     # d
    [0, 1, 2],  # q
    [0, 1],     # P
    [0, 1],     # D
    [0, 1, 2],  # Q
))
_m = 12

def _sarima_mape(tr, val, p, d, q, P, D, Q):
    """Fit SARIMA and return MAPE %, or NaN on failure."""
    try:
        res = SARIMAX(
            tr, order=(p, d, q), seasonal_order=(P, D, Q, _m),
            trend='n', enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc  = res.get_forecast(steps=len(val)).predicted_mean.values
        yt  = val.values.astype(float)
        return float(np.mean(np.abs((yt - fc) / yt)) * 100)
    except Exception:
        return np.nan


# ── Tuning 1: Walk-forward CV — fold1 (24→12) + fold2 (36→12) ────────────
# fold1: train month 1-24, validate month 25-36
# fold2: train month 1-36, validate month 37-48
tr1, val1 = train_y.iloc[:24], train_y.iloc[24:36]
tr2, val2 = train_y.iloc[:36], train_y.iloc[36:48]

rows = []
for p, d, q, P, D, Q in _grid:
    m1 = _sarima_mape(tr1, val1, p, d, q, P, D, Q)
    m2 = _sarima_mape(tr2, val2, p, d, q, P, D, Q)
    if not (np.isnan(m1) or np.isnan(m2)):
        rows.append({
            'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q,
            'MAPE_fold1': round(m1, 2),
            'MAPE_fold2': round(m2, 2),
            'MAPE_avg'  : round((m1 + m2) / 2, 2),
        })

df_tune1 = pd.DataFrame(rows).sort_values('MAPE_avg').reset_index(drop=True)
best1    = df_tune1.iloc[0]
print("Tuning 1 (Walk-forward CV) — Best Params:")
print(f"  order=({int(best1.p)},{int(best1.d)},{int(best1.q)})  "
      f"seasonal_order=({int(best1.P)},{int(best1.D)},{int(best1.Q)},12)  "
      f"MAPE={best1.MAPE_avg:.2f}%")
print("\nTop 5:")
display(df_tune1.head(5))



##############################################


# ── Tuning 2: 3-year train → 1-year validation ────────────────────────────
# train month 1-36, validate month 37-48
tr, val = train_y.iloc[:36], train_y.iloc[36:48]

rows = []
for p, d, q, P, D, Q in _grid:
    mape = _sarima_mape(tr, val, p, d, q, P, D, Q)
    if not np.isnan(mape):
        rows.append({'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 'MAPE': round(mape, 2)})

df_tune2 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
best2    = df_tune2.iloc[0]
print("Tuning 2 (3yr train → 1yr val) — Best Params:")
print(f"  order=({int(best2.p)},{int(best2.d)},{int(best2.q)})  "
      f"seasonal_order=({int(best2.P)},{int(best2.D)},{int(best2.Q)},12)  "
      f"MAPE={best2.MAPE:.2f}%")
print("\nTop 5:")
display(df_tune2.head(5))


##########################################################

# ── Tuning 3: 2-year train → 1-year validation ────────────────────────────
# train month 1-24, validate month 25-36
tr, val = train_y.iloc[:24], train_y.iloc[24:36]

rows = []
for p, d, q, P, D, Q in _grid:
    mape = _sarima_mape(tr, val, p, d, q, P, D, Q)
    if not np.isnan(mape):
        rows.append({'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 'MAPE': round(mape, 2)})

df_tune3 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
best3    = df_tune3.iloc[0]
print("Tuning 3 (2yr train → 1yr val) — Best Params:")
print(f"  order=({int(best3.p)},{int(best3.d)},{int(best3.q)})  "
      f"seasonal_order=({int(best3.P)},{int(best3.D)},{int(best3.Q)},12)  "
      f"MAPE={best3.MAPE:.2f}%")
print("\nTop 5:")
display(df_tune3.head(5))

##########################################################


# ── Tuning 4: Full training set → test set (real out-of-sample) ──────────
# train on all of train_y, validate on test_y
rows = []
for p, d, q, P, D, Q in _grid:
    mape = _sarima_mape(train_y, test_y, p, d, q, P, D, Q)
    if not np.isnan(mape):
        rows.append({'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 'MAPE': round(mape, 2)})

df_tune4 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
best4    = df_tune4.iloc[0]
print("Tuning 4 (Full train → test set) — Best Params:")
print(f"  order=({int(best4.p)},{int(best4.d)},{int(best4.q)})  "
      f"seasonal_order=({int(best4.P)},{int(best4.D)},{int(best4.Q)},12)  "
      f"MAPE={best4.MAPE:.2f}%")
print("\nTop 5:")
display(df_tune4.head(5))
