# Convert to string for safe comparison — avoids datetime.date & operator bug
fc_str = set(str(d) for d in forecast_dates)
ref['DATE'] = ref['DATE'].apply(lambda x: datetime.date.fromisoformat(str(x)))
ref = ref[ref['DATE'].apply(lambda x: str(x) in fc_str)]



# Normalise both indexes to plain strings → no datetime.date type conflicts
pred_s = pd.Series(preds_array,
                   index=[str(d) for d in forecast_dates])
test_s = test_series.copy()
test_s.index = [str(d) for d in test_s.index]

common = pred_s.index.intersection(test_s.index)

if len(common) > 0:
    test_mape = safe_mape(test_s.loc[common].values,
                           pred_s.loc[common].values)
else:
    test_mape = np.nan





# In actual_rows — convert index to date strings for consistency
actual_rows = pd.DataFrame({
    'DATE'         : [str(d) for d in test_series.index],
    'METRIC_VALUE' : list(test_series.values),
    'FORECAST_TYPE': 'Actual'
})

# In pred_rows
pred_rows = pd.DataFrame({
    'DATE'         : [str(d) for d in forecast_dates],
    'METRIC_VALUE' : list(preds_array),
    'FORECAST_TYPE': model_name
})

# In ref (2025 0+12)
ref['DATE'] = ref['DATE'].apply(lambda x: str(datetime.date.fromisoformat(str(x))))
