# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ----------- LOAD DATA -----------
df = pd.read_csv("creditcard.csv") #change filename if needed

# ----------- EDA -----------
print("HEAD:\n", df.head())
print("\nINFO:")
print(df.info())
print("\nDESCRIBE:\n", df.describe())

print("\nMISSING VALUES BEFORE:\n", df.isnull().sum())

# ----------- PREPROCESSING -----------

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nDATA AFTER PREPROCESSING:\n", df.head())

print("\nMISSING VALUES AFTER:\n", df.isnull().sum())

# Keep only numeric columns
df = df.select_dtypes(include=np.number)

# ----------- NORMALIZATION -----------
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(df)

print("\nNORMALIZED DATA SAMPLE")
print(normalized_data[:5])

# ----------- SCALING -----------
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

print("\nSCALED DATA SAMPLE")
print(scaled[:5])

# ----------- DATA VISUALIZATION -----------
cols = df.columns[:6]

# Histogram
df[cols].hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Boxplot
for col in cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.show()
