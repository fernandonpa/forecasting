import itertools

# Simple grid
_grid = list(itertools.product(
    [0, 1, 2],  # p
    [0, 1],     # d
    [0, 1, 2],  # q
    [0, 1],     # P
    [0, 1],     # D
    [0, 1, 2],  # Q
))

# IMPORTANT:
# This tuning uses the SAME objects/logic as the model cell:
# - train_y_fit, exog_train
# - recursive_forecast_with_exog(...)
# - mean_absolute_percentage_error(...)
# So selected hyperparameters should reproduce the same MAPE when reapplied.

rows = []
for p, d, q, P, D, Q in _grid:
    try:
        candidate = SARIMAX(
            endog=train_y_fit,
            exog=exog_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 12),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # Full test set recursive forecast (all test points)
        y_pred = recursive_forecast_with_exog(
            result_obj=candidate,
            history=train_y_fit,
            future_index=test_y.index,
        ).values.astype(float)
        y_true = test_y.values.astype(float)

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        rows.append({
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q,
            'MAPE': round(float(mape), 4),
            'n_test_points': int(len(test_y)),
        })
    except Exception:
        continue

if len(rows) == 0:
    print('No valid parameter combinations were fitted. Try reducing the grid.')
    df_tune4 = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'MAPE', 'n_test_points'])
else:
    df_tune4 = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
    best4 = df_tune4.iloc[0]
    print('Tuning 4 (Full train -> full test, same recursive path) - Best Params:')
    print(f"  order=({int(best4.p)},{int(best4.d)},{int(best4.q)})  "
          f"seasonal_order=({int(best4.P)},{int(best4.D)},{int(best4.Q)},12)  "
          f"MAPE={best4.MAPE:.4f}%")

print('\nTop 5:')
display(df_tune4.head(5))
