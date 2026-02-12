import pandas as pd
import numpy as np

df = pd.read_csv("ml.csv")

# Fill missing numeric values
num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print(df.info())

# Label Encoding (store inside df)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Gender_Encoded"] = le.fit_transform(df["Gender"])

# One Hot Encoding (overwrite df)
df = pd.get_dummies(df, columns=["City", "Department"])

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())

# Now store it in 
df.to_pickle("cleaned_scaled.pkl")