#logistic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df =pd.read_pickle("cleaned_scaled.pkl")

X = df[["Age","Income"]]
y = df["Bought_Encoded"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

logr=LogisticRegression(max_iter=1000)
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

print(logr.score(X_test,y_test))

# Create mesh grid
x_min, x_max = X["Age"].min() - 1, X["Age"].max() + 1
y_min, y_max = X["Income"].min() - 1, X["Income"].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Predict for grid points
grid = np.c_[xx.ravel(), yy.ravel()]
Z = logr.predict(grid)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot actual data points
plt.scatter(X_test["Age"], X_test["Income"], c=y_test)

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Logistic Regression Decision Boundary")

plt.show()