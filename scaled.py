#scaled.py
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

df["Bought_Encoded"] = le.fit_transform(df["Target_Bought"])

# One Hot Encoding (overwrite df)
df = pd.get_dummies(df, columns=["City", "Department"])

from sklearn.preprocessing import StandardScaler

  # Scaling

scaler = StandardScaler()

# Do NOT scale target column
target_col = "Bought_Encoded"

num_cols = df.select_dtypes(include="number").columns
num_cols = num_cols.drop(target_col)  # remove target

df[num_cols] = scaler.fit_transform(df[num_cols])


print(df.head())

# Now store it in 
df.to_pickle("cleaned_scaled.pkl")