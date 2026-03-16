# Cell 1: Imports
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress minor warnings for a cleaner notebook
warnings.filterwarnings('ignore')


########################

# Cell 2: Configuration
FILE_PATH = 'plot.xlsx'

# Define colors based on column position (0 is the Date column)
# 1 = Column B, 2 = Column C, 3 = Column D, etc.
COLOR_INDEX_PALETTE = {
    1: '#A2CEED',   # e.g., Actual
    2: 'green',     # e.g., original base case
    3: 'orange',    # e.g., 2025 0+12
    4: 'red',       # e.g., Best Statistical Model
    5: 'purple',    # e.g., Best uni ML Model
    6: 'brown',     # e.g., Best Chronos
    7: '#FF5722',   # e.g., Best ml ML Model
    8: 'cyan'       # Add more index numbers if you have more columns
}

##############################3

# Cell 3: Plotting Function
def plot_results(pred_df: pd.DataFrame, metric: str, palette: dict) -> None:
    existing = [k for k in pred_df['FORECAST_TYPE'].unique()]
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(metric, fontsize=14, fontweight='bold')
    
    for label in existing:
        subset = pred_df[pred_df['FORECAST_TYPE'] == label].sort_values('DATE')
        ax.plot(
            range(len(subset)),
            subset['METRIC_VALUE'].values,
            marker=None, 
            label=label,
            color=palette.get(label, None), 
            linewidth=2
        )
        
    x_labels = (
        pred_df[pred_df['FORECAST_TYPE'] == existing[0]]
        .sort_values('DATE')['DATE'].astype(str).tolist()
    )
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_xlabel('DATE')
    ax.set_ylabel('METRIC_VALUE')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.show()


#############################
# Cell 4: Read, Process, and Plot Data
xls = pd.ExcelFile(FILE_PATH)

for sheet_name in xls.sheet_names:
    print(f"--- Generating Plot for Tab: {sheet_name} ---")
    
    # 1. Read raw data
    raw_df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 2. Handle headers if they start on row 2
    if 'Date' not in raw_df.columns and 'DATE' not in raw_df.columns:
        date_row_mask = raw_df.isin(['Date', 'DATE']).any(axis=1)
        if date_row_mask.any():
            header_idx = date_row_mask.idxmax()
            raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=header_idx + 1)
            
    raw_df.dropna(how='all', axis=1, inplace=True)
    raw_df.dropna(how='all', axis=0, inplace=True)

    date_col = [col for col in raw_df.columns if str(col).lower().strip() == 'date']
    if not date_col:
        print(f"Skipping '{sheet_name}' - No 'Date' column found.")
        continue
        
    date_col_name = date_col[0]
    
    # --- NEW: Build dynamic color palette for this specific tab ---
    dynamic_palette = {}
    current_columns = list(raw_df.columns)
    
    for idx, col_name in enumerate(current_columns):
        if idx in COLOR_INDEX_PALETTE:
            dynamic_palette[col_name] = COLOR_INDEX_PALETTE[idx]
    # --------------------------------------------------------------
    
    # 3. Reshape from Wide format to Long format
    melted_df = raw_df.melt(
        id_vars=[date_col_name],
        var_name='FORECAST_TYPE',
        value_name='METRIC_VALUE'
    )
    
    melted_df.rename(columns={date_col_name: 'DATE'}, inplace=True)
    melted_df['METRIC_VALUE'] = pd.to_numeric(melted_df['METRIC_VALUE'], errors='coerce')
    melted_df.dropna(subset=['METRIC_VALUE'], inplace=True)
    
    # 4. Plot the data using the dynamically generated palette
    plot_results(
        pred_df=melted_df, 
        metric=f"Forecast Results: {sheet_name}", 
        palette=dynamic_palette
    )
