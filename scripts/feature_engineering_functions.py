import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

def create_aggregate_features(df):
    """
    Create aggregated transaction features for each customer.
    """
    df['total_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['average_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['transaction_count'] = df.groupby('CustomerId')['TransactionId'].transform('count')
    df['std_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('std')
    
    return df

def extract_datetime_features(df):
    """
    Extract datetime features from the TransactionStartTime column.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    
    return df

def encode_categorical_variables(df, categorical_columns, encoding_type="onehot"):
    """
    Encode categorical columns using one-hot or label encoding.
    """
    missing_columns = [col for col in categorical_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"The following columns are missing: {missing_columns}")
    
    if encoding_type == "onehot":
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    elif encoding_type == "label":
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[f'{col}_encoded'] = label_encoder.fit_transform(df[col].astype(str))
    return df

def handle_missing_values(df, strategy="imputation"):
    """
    Handle missing values by imputation or removal.
    """
    if strategy == "imputation":
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(0, inplace=True)  # Fill numerical columns with 0
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical columns with mode
    elif strategy == "removal":
        df.dropna(inplace=True)
    return df

def scale_numerical_features(df, numerical_columns, scaling_type="standardize"):
    """
    Scale numerical columns using standardization or normalization.
    """
    if scaling_type == "normalize":
        scaler = MinMaxScaler()
    elif scaling_type == "standardize":
        scaler = StandardScaler()
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

def bin_numerical_features(df, numerical_columns, num_bins=10, method="quantile"):
    """
    Bin continuous numerical features into categories.
    Args:
        df: DataFrame containing the data.
        numerical_columns: List of columns to be binned.
        num_bins: Number of bins for binning the features.
        method: 'quantile' for quantile-based binning or 'uniform' for uniform binning.
    Returns:
        DataFrame with binned features.
    """
    for col in numerical_columns:
        if method == "quantile":
            df[f'{col}_bin'] = pd.qcut(df[col], q=num_bins, duplicates='drop')
        elif method == "uniform":
            df[f'{col}_bin'] = pd.cut(df[col], bins=num_bins)
    
    return df

def feature_engineer_rfms(data):
    """
    Feature engineering to create Recency, Frequency, and Monetary columns for RFMS analysis.
    Additionally, a Severity column is created based on the monetary value.
    """
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    current_date = data['TransactionStartTime'].max()
    
    # Calculate Recency, Frequency, and Monetary
    data['Recency'] = data.groupby('CustomerId')['TransactionStartTime'].transform(lambda x: (current_date - x.max()).days)
    data['Frequency'] = data['transaction_count']
    data['Monetary'] = data['total_transaction_amount']
    
    # Calculate Severity based on Monetary (for example: high, medium, low)
    bins = [0, 100, 500, 1000, float('inf')]  # Define your bins
    labels = ['Low', 'Medium', 'High', 'Very High']  # Define your labels
    data['Severity'] = pd.cut(data['Monetary'], bins=bins, labels=labels, right=False)
    
    # Remove duplicate transactions if necessary
    data.drop_duplicates(subset=['TransactionId'], inplace=True)
    
    return data

def feature_engineering_pipeline(df, categorical_columns):
    """
    Full feature engineering pipeline.
    """
    df = create_aggregate_features(df)
    df = extract_datetime_features(df)
    df = handle_missing_values(df)
    df = feature_engineer_rfms(df)
    
    # Define numerical columns for scaling and binning
    numerical_columns = ['total_transaction_amount', 'average_transaction_amount', 
                        'transaction_count', 'std_transaction_amount', 
                        'Recency', 'Frequency', 'Monetary']
    
    # Scale numerical features
    df = scale_numerical_features(df, numerical_columns, scaling_type="standardize")
    
    # Bin numerical features if necessary
    df = bin_numerical_features(df, numerical_columns, num_bins=5, method="quantile")

    # Encode categorical variables
    df = encode_categorical_variables(df, categorical_columns)
    
    return df

# Example usage
# df = pd.read_csv('your_data.csv')  # Load your data
# categorical_columns = ['CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId']  # Define your categorical columns
# engineered_df = feature_engineering_pipeline(df, categorical_columns)
