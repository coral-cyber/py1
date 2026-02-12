import pandas as pd
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