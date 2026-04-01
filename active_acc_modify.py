# 1. Define the 12-month future index
index_2026 = pd.date_range(start='2026-01-31', end='2026-12-31', freq='ME')

# 2. Create a new exogenous dataframe specifically for these 12 future months. 
# It applies the same logic you used for train/test exog.
future_exog = pd.DataFrame(
    {'Portfolio_Acquired': (index_2026 <= step_date).astype(float)}, # ensure type matches your train_exog
    index=index_2026
)

# 3. Request 12 steps and pass the new 12-month future_exog
# (Remember to append .predicted_mean to extract the series)
acc_fc_obj = acc_result.get_forecast(steps=len(index_2026), exog=future_exog)
acc_fc = acc_fc_obj.predicted_mean
acc_fc.index = index_2026

# 4. Safely join the 2-month actuals with the 12-month forecast
acc_compare = pd.concat({
    'actual_active_accounts': acc_test_y,
    'forecast_active_accounts': acc_fc
}, axis=1)

display(acc_compare.head(12))
