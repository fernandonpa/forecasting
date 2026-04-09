train_y = train_df[TARGET_METRIC].astype(float)
test_y  = test_df[TARGET_METRIC].astype(float)


def build_exog_from_series(y: pd.Series) -> pd.DataFrame:
    """Feature set requested: lag_1, lag_6, lag_12, month_num, ma_3."""
    idx = pd.to_datetime(y.index)
    exog = pd.DataFrame(index=y.index)
    exog['lag_1'] = y.shift(1)
    exog['lag_6'] = y.shift(6)
    exog['lag_12'] = y.shift(12)
    exog['ma_3'] = y.shift(1).rolling(3).mean()
    exog['month_num'] = idx.month.astype(int)
    return exog


# Build training exog and align after lag/MA NaNs
exog_train_full = build_exog_from_series(train_y)
valid_idx = exog_train_full.dropna().index
train_y_fit = train_y.loc[valid_idx]
exog_train = exog_train_full.loc[valid_idx]

sarimax_model = SARIMAX(
    endog=train_y_fit,
    exog=exog_train,
    order=(1, 1, 0),
    seasonal_order=(0, 1, 1, 12),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarimax_result = sarimax_model.fit(disp=False)
print(sarimax_result.summary())


# Recursive forecast helper:
# each new prediction is appended and reused for lag_1 / lag_6 / lag_12 / ma_3 of the next step.
def recursive_forecast_with_exog(result_obj, history: pd.Series, future_index) -> pd.Series:
    preds = []
    state = result_obj
    hist = history.copy()

    for dt in future_index:
        lag_1 = float(hist.iloc[-1])
        lag_6 = float(hist.iloc[-min(len(hist), 6)])
        lag_12 = float(hist.iloc[-min(len(hist), 12)])
        ma_3 = float(hist.iloc[-3:].mean())
        month_num = int(pd.to_datetime(dt).month)

        x_next = pd.DataFrame(
            {'lag_1': [lag_1], 'lag_6': [lag_6], 'lag_12': [lag_12], 'ma_3': [ma_3], 'month_num': [month_num]},
            index=[dt],
        )

        y_hat = float(state.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0])
        preds.append(y_hat)

        y_new = pd.Series([y_hat], index=[dt])
        hist = pd.concat([hist, y_new])
        state = state.append(endog=y_new, exog=x_next, refit=False)

    return pd.Series(preds, index=future_index)


# 1) Test-period recursive predictions
sarimax_fc_test = recursive_forecast_with_exog(
    result_obj=sarimax_result,
    history=train_y_fit,
    future_index=test_y.index,
)

pred_df_sarimax = build_pred_df(sarimax_fc_test.values, test_y)
pred_df_sarimax.loc[
    pred_df_sarimax['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'
] = 'SARIMAX_exog_recursive'
pred_df_sarimax


#############################

plot_results(pred_df_sarimax, f'SARIMAX + exog (lag_1, lag_6, lag_12, ma_3, month_num) — {TARGET_METRIC}')
evaluate_pred_df(pred_df_sarimax, 'SARIMAX_exog_recursive')


# 2) Future-year recursive forecast example: 2026 (12 months)
last_actual_date = pd.to_datetime(train_y.index.max())
future_2026_idx = pd.date_range('2026-01-31', periods=12, freq='M').date

# Refit on all currently available Actual history before forecasting 2026
full_actual_y = data_df[TARGET_METRIC].dropna().astype(float)
full_exog = build_exog_from_series(full_actual_y)
full_valid_idx = full_exog.dropna().index
full_y_fit = full_actual_y.loc[full_valid_idx]
full_exog_fit = full_exog.loc[full_valid_idx]

sarimax_full = SARIMAX(
    endog=full_y_fit,
    exog=full_exog_fit,
    order=(1, 1, 0),
    seasonal_order=(0, 1, 1, 12),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

fc_2026 = recursive_forecast_with_exog(
    result_obj=sarimax_full,
    history=full_y_fit,
    future_index=future_2026_idx,
)

future_2026_df = pd.DataFrame({
    'DATE': future_2026_idx,
    'FORECAST_TYPE': 'SARIMAX_exog_recursive_2026',
    'METRIC_VALUE': fc_2026.values,
})

print('2026 recursive forecast (first 5 rows):')
display(future_2026_df.head())

plt.figure(figsize=(14, 4))
plt.plot(future_2026_df['DATE'].astype(str), future_2026_df['METRIC_VALUE'], marker='o', color='#FF5722')
plt.title(f'2026 Forecast using recursive lag_1 / lag_6 / lag_12 / ma_3 / month_num — {TARGET_METRIC}')
plt.xlabel('DATE')
plt.ylabel('METRIC_VALUE')
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.show()
