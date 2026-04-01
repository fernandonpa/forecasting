def build_pred_df(forecast_values, index, test_y=None):
    """"Combine Actual, Prediction, and benchmarks into a long-format DataFrame."""
    pred_rows = pd.DataFrame({
        'DATE' : index,
        'FORECAST_TYPE': 'Prediction',
        'METRIC_VALUE' : forecast_values,
    })
    
    bench_rows_1 = get_benchmark(index, '2026 0+12')
    bench_rows_2 = get_benchmark(index, 'original base case')
    
    dfs = [
        pred_rows, 
        bench_rows_1[['DATE', 'FORECAST_TYPE', 'METRIC_VALUE']], 
        bench_rows_2[['DATE', 'FORECAST_TYPE', 'METRIC_VALUE']]
    ]
    
    # NEW: Append Actuals so they exist for plotting and metric evaluation
    if test_y is not None:
        actual_rows = pd.DataFrame({
            'DATE': test_y.index,
            'FORECAST_TYPE': 'Actual',
            'METRIC_VALUE': test_y.values
        })
        dfs.append(actual_rows)

    pred_df = pd.concat(dfs, ignore_index=True)
    pred_df['DATE'] = pred_df['DATE'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)
    return pred_df


######################333


def plot_results(pred_df: pd.DataFrame, metric: str) -> None:
    palette = {'Actual': '#A2CEED', 'SARIMAX':'orange', 'original base case': 'green', 'Statistical Model prediction' : 'red', 'SARIMA': 'orange', 'Holt-Winters':'#FF5722', 'ETS':'#FF5722', '2026 0+12': 'yellow'}
    order = ['Actual', 'SARIMAX', 'Statistical Model prediction', 'original base case', 'SARIMA', 'Holt-Winters', 'ETS', '2026 0+12']
    existing = [k for k in order if k in pred_df['FORECAST_TYPE'].unique()]

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title(metric, fontsize=14, fontweight='bold')

    # NEW: Create a uniform, sorted list of all unique dates
    pred_df['DATE_STR'] = pred_df['DATE'].astype(str)
    all_dates = sorted(pred_df['DATE_STR'].unique())

    for label in existing:
        subset = pred_df[pred_df['FORECAST_TYPE'] == label].sort_values('DATE_STR')
        
        # NEW: Find the exact x-axis index for each point's date so varying lengths align perfectly
        x_indices = [all_dates.index(d) for d in subset['DATE_STR']]
        
        ax.plot(
            x_indices,
            subset['METRIC_VALUE'].values,
            marker='o', # Added marker so your 2-month actuals are clearly visible as dots
            label=label,
            color=palette.get(label, None), 
            linewidth=2,
        )

    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=90)
    ax.set_xlabel('DATE')
    ax.set_ylabel('METRIC_VALUE')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.5f}'))
    plt.tight_layout()
    plt.show()

######################################33

# Create the 12-month index for 2026
index_2026 = pd.date_range(start='2026-01-31', end='2026-12-31', freq='ME')

# 1) Test-period recursive predictions (12 months)
sarimax_fc_test = recursive_forecast_with_exog(
    result_obj=sarimax_result,
    history=train_y_fit,
    future_index=index_2026,
)

# 2) Build DF, explicitly passing `test_y` (which only has 2 months)
pred_df_sarimax = build_pred_df(sarimax_fc_test.values, index_2026, test_y=test_y)

# 3) Rename Prediction
pred_df_sarimax.loc[
    pred_df_sarimax['FORECAST_TYPE'] == 'Prediction', 'FORECAST_TYPE'
] = "Statistical Model prediction"

# 4) Save to CSV
predictions = pred_df_sarimax[pred_df_sarimax["FORECAST_TYPE"]=="Statistical Model prediction"]
Values = predictions["METRIC_VALUE"]
Values.to_csv("best_loss_rate_lp_with_lags.csv", index=False)

# 5) Plot and Evaluate (evaluate_pred_df will automatically handle the 2-month overlap)
plot_results(pred_df_sarimax, f'SARIMA - {TARGET_METRIC}')
evaluate_pred_df(pred_df_sarimax, "Statistical Model prediction")

