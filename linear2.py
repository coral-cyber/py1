import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_pickle("cleaned_unscaled.pkl")


X = df[["Age", "SpendingScore"]]
y = df["Income"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print("\nFirst 5 Real Income Predictions:")
print(y_pred[:5])


print(lr.score(X_test,y_test))

# Scatter plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)

# Perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Actual vs Predicted Income")

plt.show()