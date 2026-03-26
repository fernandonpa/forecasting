import itertools
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Optional: suppress statsmodels convergence warnings during grid search to keep the console clean
warnings.filterwarnings("ignore") 

# Your original grid structure
_grid = list(itertools.product(
    [0, 1, 2],       # p
    [0, 1],          # d
    [0, 1],          # q
    [0, 1, 2],       # P
    [0, 1],          # D
    [0, 1]           # Q
))

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
            history_y=train_y_fit,
            future_index=test_y.index,
            acc_future=acc_test,
        ).values.astype(float)
        
        y_true = test_y.values.astype(float)

        # ---------------------------------------------------------
        # NEW: SHAPE-AWARE SCORING LOGIC
        # ---------------------------------------------------------
        base_mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        try:
            # Measure correlation to ensure the model captures seasonal peaks
            corr, _ = pearsonr(y_true, y_pred)
            corr = max(0, corr) # Treat negative correlation as 0
        except:
            corr = 0.0 # Catch completely flat-line predictions
            
        # Apply the penalty: Flat lines (corr=0) inflate the MAPE score by 1.5x
        shape_penalty_multiplier = 1 + (0.5 * (1 - corr))
        custom_score = base_mape * shape_penalty_multiplier
        # ---------------------------------------------------------

        rows.append({
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q,
            'Custom_Score': round(float(custom_score), 4), # Primary ranking metric
            'Raw_MAPE': round(float(base_mape), 4),        # Kept for reference
            'Correlation': round(float(corr), 4),          # Kept for reference
            'n_test_points': int(len(test_y)),
        })
        
    except Exception as e:
        # print(e)  # Keep commented out unless you need to debug specific failures
        continue

# ---------------------------------------------------------
# UPDATED EVALUATION BLOCK
# ---------------------------------------------------------
if len(rows) == 0:
    print('No valid parameter combinations were fitted. Try reducing the grid.')
    df_tune4 = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'Custom_Score', 'Raw_MAPE', 'Correlation', 'n_test_points'])
else:
    # IMPORTANT: Sort by Custom_Score instead of Raw_MAPE
    df_tune4 = pd.DataFrame(rows).sort_values('Custom_Score').reset_index(drop=True)
    best4 = df_tune4.iloc[0]
    
    print('Tuning 4 (Full train -> full test, same recursive path) - Best Params:')
    print(f"  order=({int(best4.p)},{int(best4.d)},{int(best4.q)})  "
          f"seasonal_order=({int(best4.P)},{int(best4.D)},{int(best4.Q)},12)  ")
    print(f"  Custom_Score={best4.Custom_Score:.4f} (Raw MAPE: {best4.Raw_MAPE:.4f}, Correlation: {best4.Correlation:.4f})")

print('\nTop 5:')
display(df_tune4.head(5))
