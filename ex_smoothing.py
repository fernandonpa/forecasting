from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Identify the jump dates
# Assuming July 2022 is the last month of the old regime, and August 2022 is the new regime
last_old_month = '2022-07-31'
first_new_month = '2022-08-31'

# 2. Calculate the exact dollar amount of the acquisition jump
val_before = train_y.loc[last_old_month]
val_after  = train_y.loc[first_new_month]
jump_amount = val_after - val_before

# 3. Create a Stitched Dataset (Retrospective Leveling)
train_y_stitched = train_y.copy()

# Add the jump amount to ALL data points before the acquisition
# This slides the old history up to match the new baseline
train_y_stitched.loc[train_y_stitched.index <= last_old_month] += jump_amount

# 4. Train Holt-Winters on the smooth, stitched data
hw_model = ExponentialSmoothing(
    train_y_stitched,         # <-- Feed the stitched data, not the raw data
    trend='add',              # <-- CHANGED: 'add' is much safer for linear downward slopes
    seasonal='add',
    seasonal_periods=12
)
hw_result = hw_model.fit(optimized=True)
print(hw_result.summary())

# 5. Generate the Forecast
# (No manual adjustment needed here because the forecast naturally projects from the new, higher baseline)
hw_fc = hw_result.forecast(steps=len(test_y))
hw_fc.index = test_y.index

# Build prediction dataframe
pred_df_hw = build_pred_df(hw_fc.values, test_y)
pred_df_hw.loc[pred_df_hw['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'] = 'Holt-Winters'
