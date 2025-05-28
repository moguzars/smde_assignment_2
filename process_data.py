import pandas as pd
from scipy import stats

data = pd.read_csv('Walmart_Sales.csv')

# process date
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Week'] = data['Date'].dt.isocalendar().week
data['Month'] = data['Date'].dt.month

# Handling outliers sales
data['Weekly_Sales'] = stats.mstats.winsorize(data['Weekly_Sales'], limits=[0.05, 0.05])

# Feature Binning
def bin_variable(df, col, bins=3, labels=['Low', 'Medium', 'High']):
    df[f'{col}_Bin'] = pd.qcut(df[col], q=bins, labels=labels)
    return df

for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
    data = bin_variable(data, col)


data.to_csv('Walmart_Sales_processed.csv', index=False)  