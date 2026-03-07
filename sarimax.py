import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure index is datetime so we can filter mathematically
train_df.index = pd.to_datetime(train_df.index)
test_df.index = pd.to_datetime(test_df.index)

# Create the Step Dummy: 0 before the jump, 1 after the jump
train_df['Portfolio_Acquired'] = (train_df.index > '2022-07-31').astype(int)
test_df['Portfolio_Acquired'] = (test_df.index > '2022-07-31').astype(int)

# Separate the targets (endog) and the features (exog)
train_y = train_df[TARGET_METRIC].astype(float)
train_exog = train_df[['Portfolio_Acquired']]

test_y = test_df[TARGET_METRIC].astype(float)
test_exog = test_df[['Portfolio_Acquired']]


# Initialize SARIMAX with the exogenous step dummy
sarimax_model = SARIMAX(
    endog = train_y,
    exog = train_exog,               # <-- Feed the step dummy here
    order = (1, 1, 0),               # d=1 to track the downward decay
    seasonal_order = (1, 0, 0, 12),  # P=1 to track the December spikes
    trend = 'c',                     # 'c' calculates the steady linear drop
    enforce_stationarity = False,
    enforce_invertibility = False
)

sarimax_result = sarimax_model.fit(disp=False)
print(sarimax_result.summary())  


# Generate forecast, explicitly passing the future exogenous values
sarimax_fc = sarimax_result.get_forecast(
    steps = len(test_y), 
    exog = test_exog                 # <-- Required for future prediction
).predicted_mean

sarimax_fc.index = test_y.index

# Build prediction dataframe using your custom function
pred_df_sarimax = build_pred_df(sarimax_fc.values, test_y)
pred_df_sarimax.loc[pred_df_sarimax['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'] = 'SARIMAX'
