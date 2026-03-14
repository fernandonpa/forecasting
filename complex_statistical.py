from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.varmax import VARMAX


PRE_DATA_CUTOFF = datetime.date(2023, 5, 31)

exog_train = pd.DataFrame({
    'pre_data_flag': (pd.to_datetime(train_y.index).date <= PRE_DATA_CUTOFF).astype(int)
}, index=train_y.index)
exog_test = pd.DataFrame({
    'pre_data_flag': (pd.to_datetime(test_y.index).date <= PRE_DATA_CUTOFF).astype(int)
}, index=test_y.index)

bsts_model = UnobservedComponents(
    endog=train_y,
    level='local linear trend',
    seasonal=12,
    exog=exog_train,
)
bsts_result = bsts_model.fit(disp=False)
print(bsts_result.summary())

bsts_fc = bsts_result.get_forecast(steps=len(test_y), exog=exog_test).predicted_mean
bsts_fc.index = test_y.index

pred_df_bsts = build_pred_df(bsts_fc.values, test_y)
pred_df_bsts.loc[pred_df_bsts['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'] = 'BSTS'
pred_df_bsts
#####################
plot_results(pred_df_bsts, f'BSTS (Structural + pre_data_flag) — {TARGET_METRIC}')
evaluate_pred_df(pred_df_bsts, 'BSTS')

#############

# BSTS hyperparameter tuning (MAPE on test set)
# simple grid: trend type x seasonal period

bsts_grid = [
    {'level': 'local level', 'seasonal': 12},
    {'level': 'local linear trend', 'seasonal': 12},
    {'level': 'local linear trend', 'seasonal': 6},
    {'level': 'local level', 'seasonal': 6},
]

bsts_rows = []
for params in bsts_grid:
    try:
        m = UnobservedComponents(
            endog=train_y,
            level=params['level'],
            seasonal=params['seasonal'],
            exog=exog_train,
        ).fit(disp=False)
        fc = m.get_forecast(steps=len(test_y), exog=exog_test).predicted_mean.values
        mape = mean_absolute_percentage_error(test_y.values.astype(float), fc.astype(float)) * 100
        bsts_rows.append({
            'level': params['level'],
            'seasonal': params['seasonal'],
            'MAPE': round(mape, 2),
        })
    except Exception:
        continue

df_bsts_tune = pd.DataFrame(bsts_rows).sort_values('MAPE').reset_index(drop=True)
print('BSTS tuning — Best Params:')
print(df_bsts_tune.iloc[0].to_dict())
print('\nTop 5:')
display(df_bsts_tune.head(5))


###########################

endog_cols = list(train_df.columns)
train_endog = train_df[endog_cols].astype(float)
test_endog  = test_df[endog_cols].astype(float)

var_model = VARMAX(
    endog=train_endog,
    exog=exog_train,
    order=(1, 0),
    trend='n',
    enforce_stationarity=False,
)
var_result = var_model.fit(disp=False, maxiter=200)
print(var_result.summary())

var_fc_all = var_result.forecast(steps=len(test_endog), exog=exog_test)
var_fc = var_fc_all[TARGET_METRIC]
var_fc.index = test_y.index

pred_df_var = build_pred_df(var_fc.values, test_y)
pred_df_var.loc[pred_df_var['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'] = 'VAR'
pred_df_var

##########

plot_results(pred_df_var, f'VAR (VARX with pre_data_flag) — {TARGET_METRIC}')
evaluate_pred_df(pred_df_var, 'VAR')

########################33

# VAR hyperparameter tuning (MAPE on test set)
# tuning lag order p for VARX(p) via VARMAX(order=(p,0))

var_rows = []
for p in [1, 2, 3]:
    try:
        vm = VARMAX(
            endog=train_endog,
            exog=exog_train,
            order=(p, 0),
            trend='n',
            enforce_stationarity=False,
        ).fit(disp=False, maxiter=200)

        fc_all = vm.forecast(steps=len(test_endog), exog=exog_test)
        fc = fc_all[TARGET_METRIC].values.astype(float)
        mape = mean_absolute_percentage_error(test_y.values.astype(float), fc) * 100

        var_rows.append({'p': p, 'q': 0, 'MAPE': round(mape, 2)})
    except Exception:
        continue

df_var_tune = pd.DataFrame(var_rows).sort_values('MAPE').reset_index(drop=True)
print('VAR tuning — Best Params:')
print(df_var_tune.iloc[0].to_dict())
print('\nTop 5:')
display(df_var_tune.head(5))

