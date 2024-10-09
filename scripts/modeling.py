from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def convert_to_datetime(df, column_name):
 
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    return df

def fill_missing_values(df, column_name, method='mean'):
    
    if method == 'mean':
        df[column_name].fillna(df[column_name].mean(), inplace=True)
    elif method == 'median':
        df[column_name].fillna(df[column_name].median(), inplace=True)
    elif method == 'drop':
        df.dropna(subset=[column_name], inplace=True)
    return df

def convert_column_type(df, column_name, new_type):
   
    df[column_name] = df[column_name].astype(new_type)
    return df

def strip_strings(df, column_name):
    
    df[column_name] = df[column_name].str.strip()
    return df

def convert_to_boolean(df, column_name):
   
    df[column_name] = df[column_name].astype(bool)
    return df


def convert_to_datetime(df, column):
    df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

def fill_missing_values(df, column, method='mean'):
    if method == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    return df

def convert_to_boolean(df, column):
    df[column] = df[column].astype(bool)
    return df

def strip_strings(df, column):
    df[column] = df[column].str.strip()
    return df
def extract_datetime_features(df, timestamp_column):
   
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    
    if df[timestamp_column].isnull().any():
        print(f"Warning: {df[timestamp_column].isnull().sum()} null values in '{timestamp_column}' after conversion to datetime.")

    df[f'{timestamp_column}_year'] = df[timestamp_column].dt.year
    df[f'{timestamp_column}_month'] = df[timestamp_column].dt.month
    df[f'{timestamp_column}_day'] = df[timestamp_column].dt.day
    df[f'{timestamp_column}_hour'] = df[timestamp_column].dt.hour
    
    return df


def clean_data(df):
  
    df = convert_to_datetime(df, 'TransactionStartTime')
    df=  extract_datetime_features(df,'TransactionStartTime')
  
    df = fill_missing_values(df, 'Amount', method='mean')
    
    
    df = convert_to_boolean(df, 'ProductCategory_data_bundles')

    
    df = strip_strings(df, 'TransactionId')

    return df

def process_data(data):
  
    df = clean_data(data.copy())

    high_cardinality_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId', 'ProductId', 'Severity']
    label_encoder = LabelEncoder()

    for col in high_cardinality_columns:
        df[col] = label_encoder.fit_transform(df[col])


    low_cardinality_columns = ['CountryCode']  
    df = pd.get_dummies(df, columns=low_cardinality_columns, drop_first=True)

  
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)


    return df






def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_dist = {
        'n_estimators': range(50, 200, 10),
        'max_depth': [None] + list(range(5, 30, 5)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_random_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                                     param_dist, 
                                     n_jobs=-1, 
                                     cv=5, 
                                     scoring='accuracy')
    rf_random_search.fit(X_train, y_train)

    best_rf_model = rf_random_search.best_estimator_

    rf_pred = best_rf_model.predict(X_test)
    evaluation_metrics = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1 Score': f1_score(y_test, rf_pred),
        'ROC-AUC': roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])
    }

    return evaluation_metrics


def train_decision_tree(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_dist = {
        'max_depth': [None] + list(range(1, 20)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                                   param_dist, 
                                   n_jobs=-1, 
                                   cv=5, 
                                   scoring='accuracy')
    dt_grid_search.fit(X_train, y_train)

   
    best_dt_model = dt_grid_search.best_estimator_

    dt_pred = best_dt_model.predict(X_test)

    evaluation_metrics = {
        'Accuracy': accuracy_score(y_test, dt_pred),
        'Precision': precision_score(y_test, dt_pred),
        'Recall': recall_score(y_test, dt_pred),
        'F1 Score': f1_score(y_test, dt_pred),
        'ROC-AUC': roc_auc_score(y_test, best_dt_model.predict_proba(X_test)[:, 1])
    }

    return evaluation_metrics
