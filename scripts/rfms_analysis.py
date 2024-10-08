import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from tabulate import tabulate  

def calculate_rfms_and_woe(data: pd.DataFrame) -> pd.DataFrame:
   
    
   
    required_columns = ['Recency', 'Frequency', 'Monetary']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if data[required_columns].isnull().any().any():
        raise ValueError('One or more required columns contain NaN values after conversion.')

    data['rfms_score'] = data['Recency'] + data['Frequency'] + data['Monetary']

   
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Recency'], data['Frequency'], c=data['Monetary'], cmap='viridis', label='Monetary')
    plt.colorbar(label='Monetary Value')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.title('RFMS Scatter Plot')
    plt.axvline(data['rfms_score'].median(), color='red', linestyle='--', label='Median RFMS Score')
    plt.legend()
    plt.show()

    threshold = data['rfms_score'].median()
    data['label'] = np.where(data['rfms_score'] >= threshold, 'good', 'bad')


    binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    data['rfms_bins'] = binner.fit_transform(data[['rfms_score']]).astype(int)


    summary = []
    for bin in range(5):
        good_count = data[(data['rfms_bins'] == bin) & (data['label'] == 'good')].shape[0]
        bad_count = data[(data['rfms_bins'] == bin) & (data['label'] == 'bad')].shape[0]
      
        print(f"Bin: {bin}, Good Count: {good_count}, Bad Count: {bad_count}")
        
        if good_count == 0 or bad_count == 0:  
            woe = np.nan  
        else:
            good_prop = good_count / data[data['label'] == 'good'].shape[0]
            bad_prop = bad_count / data[data['label'] == 'bad'].shape[0]
            woe = np.log(good_prop / bad_prop)
        
        summary.append({
            'Bin': bin,
            'Good Count': good_count,
            'Bad Count': bad_count,
            'WoE': woe
        })

  
    summary_df = pd.DataFrame(summary)

   
    print("\nSummary Table:")
    print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))

  
    woe_dict = {row['Bin']: row['WoE'] for index, row in summary_df.iterrows()}
    data['woe'] = data['rfms_bins'].map(woe_dict)

    return data


