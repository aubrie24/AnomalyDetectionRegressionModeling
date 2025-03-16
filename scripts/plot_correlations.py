
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
#this script only creates plots to check the correlation between the target variables and features
#the linear regression baseline was getting a low MSE
#this is a way to investigate if the relationship is actually linear

def load_txt(data_path):
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    return data

#function to convert columns to numeric where possible
#used in pipeline preprocessing
def convert_to_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')

#the gmin column kept making the code fail, no matter if values were imputted or clipped
#name is removed because this is not included in the un published data
#used in pipeline preprocessing
def drop_problem_columns(df):
    return df.drop(columns=['gmin', 'Name'], errors='ignore')

#custom transformer to handle missing values
#this transformer will drop any columns that has a high percentage of missing values
class MissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_drop = []
    
    def fit(self, X, y=None):
        #identify columns to drop based on missing ratio
        missing_ratio = X.isnull().mean()
        self.columns_to_drop = missing_ratio[missing_ratio > self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        #drop the identified columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        X = X.fillna(X.mean())
        return X
    
#custom transformer to cap infinite values
class CapInfiniteValues(BaseEstimator, TransformerMixin):
    def __init__(self, cap_value=1e9):
        self.cap_value = cap_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], [self.cap_value, -self.cap_value])
        return X

#function to plot correlations and see if they're linear
def plot_correlation_heatmap(df, title="Correlation Heatmap", save_path=None):
    correlation = df.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation)), correlation.columns)
    
    # Annotate correlation coefficients
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.title(title)
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def main():
    output_path = "../output/results.txt"
    model_path = "../output/modeling_pipeline.pkl"
    importance_path = "../output"
    predictions_path = "../output/ranked_predictions.csv"

    # Data paths
    published_data_path = "/deac/csc/classes/csc373/data/assignment_4/published_screen.txt"

    # Load the published dataset
    published_data = load_txt(published_data_path)

    # Preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('drop_problem_columns', FunctionTransformer(drop_problem_columns)),
        ('convert_numeric', FunctionTransformer(convert_to_numeric)),
        ('handle_missing', MissingValues()),
        ('cap_infinite', CapInfiniteValues())
    ])
    
    # Apply preprocessing
    processed_data = preprocessing_pipeline.fit_transform(published_data)

    # Save plot to file
    plot_correlation_heatmap(
        processed_data,
        title="Correlation Heatmap (Processed Data)",
        save_path="../output/correlation_heatmap.png"
    )

if __name__ == "__main__":
    main()