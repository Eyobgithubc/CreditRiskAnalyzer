import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


def create_aggregate_features(df):
  
    df['total_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    
    df['average_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    
    df['transaction_count'] = df.groupby('CustomerId')['TransactionId'].transform('count')

    df['std_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('std')
    
    return df


def extract_datetime_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    
    return df

def encode_categorical_variables(df, categorical_columns, encoding_type="onehot"):
   
    missing_columns = [col for col in categorical_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"The following columns are missing: {missing_columns}")
    
    if encoding_type == "onehot":
      
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    elif encoding_type == "label":
        
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df



def handle_missing_values(df, strategy="imputation"):
    if strategy == "imputation":
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(0, inplace=True)  
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)  
    elif strategy == "removal":
        df.dropna(inplace=True)
    return df



def scale_numerical_features(df, numerical_columns, scaling_type="standardize"):
    if scaling_type == "normalize":
    
        scaler = MinMaxScaler()
    elif scaling_type == "standardize":
     
        scaler = StandardScaler()
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df
