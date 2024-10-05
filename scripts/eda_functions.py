import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    return pd.read_csv(filepath)

def data_overview(df):
    print("Shape of dataset:", df.shape)
    print("Data types:\n", df.dtypes)

def summary_statistics(df):
    summary = df.describe()
    print("### Summary Statistics\n")
    print(summary.to_markdown())

def plot_histograms(df):
    columns = ['Amount', 'Value', 'PricingStrategy']
    for col in columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def plot_box_plots(df):
    columns = ['Amount', 'Value']
    for col in columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Box Plot for {col} (Outlier Detection)')
        plt.xlabel(col)
        plt.show()

def plot_count_plots(df):
    categorical_columns = ['ProductCategory', 'ChannelId', 'CurrencyCode', 'FraudResult']
    for col in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=df)
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

def plot_correlation_heatmap(df):
    numerical_columns = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
    corr = df[numerical_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap for Numerical Features")
    plt.show()


