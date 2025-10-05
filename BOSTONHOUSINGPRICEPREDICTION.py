# Boston Housing Price Prediction using Linear Regression

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('BostonHousing.csv')

# Basic inspection
print("First 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDataset shape:", df.shape)

# Visualize missing values
plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# Correlation matrix
print("\nCorrelation matrix:")
print(df.corr())

print("\nDataset info:")
print(df.info())

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature and target separation
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeature shape:", x.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Impute missing values (if any)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train Linear Regression model
linear = LinearRegression()
linear.fit(X_train, y_train)

# Predict
y_pred = linear.predict(X_test)

# Evaluation function
def accuracy(y_test, y_pred):
    d1 = y_test - y_pred
    d2 = y_test - y_test.mean()
    r2 = 1 - (d1.dot(d1) / d2.dot(d2))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    return r2, mse, rmse, mae

# Evaluate model
r2, mse, rmse, mae = accuracy(y_test, y_pred)

print("\nModel Evaluation:")
print("R² Score:", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
