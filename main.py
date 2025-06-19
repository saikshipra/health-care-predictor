import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def main():
    # 1. Load dataset - replace 'insurance.csv' with your dataset path
    df = pd.read_csv('insurance.csv')

    # 2. Print columns to confirm dataset structure
    print("Columns in dataset:", df.columns.tolist())

    # 3. Separate features and target variable
    target_col = 'charges'  # Changed from 'expenses' to 'charges'

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 4. Split data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 5. Identify categorical columns - adjust based on your dataset
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # 6. Preprocessing: OneHotEncode
plt.savefig('healthcare_cost_prediction.png', dpi=300)
plt.show()
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")

plt.savefig('test_plot.png', dpi=300)
plt.show()

plt.savefig('healthcare_cost_prediction.png', dpi=300)  # Save before show
plt.show()
