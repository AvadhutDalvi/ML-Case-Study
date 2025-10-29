# 🏠 House Price Prediction using Linear, Ridge, and Polynomial Regression

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 📥 Load Dataset
data = pd.read_csv("house_prices.csv")  # Replace with your dataset path
print("Dataset Preview:\n", data.head())

# Selecting relevant features
X = data[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = data['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🧠 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 🔗 Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 📈 Polynomial Regression (degree = 2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# 📊 Evaluation Function
def evaluate(model_name, y_true, y_pred):
    print(f"\nModel: {model_name}")
    print("R² Score:", round(r2_score(y_true, y_pred), 3))
    print("RMSE:", round(mean_squared_error(y_true, y_pred, squared=False), 3))

# Evaluate all models
evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Ridge Regression", y_test, y_pred_ridge)
evaluate("Polynomial Regression", y_test, y_pred_poly)

# 🔍 Visualization
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.6)
plt.scatter(y_test, y_pred_ridge, label='Ridge Regression', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial Regression', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', lw=2, linestyle='--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices: Model Comparison")
plt.legend()
plt.show()
