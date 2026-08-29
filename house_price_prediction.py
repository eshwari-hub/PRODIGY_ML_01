import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")

# Select required columns
data = df[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]]

# Features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display predictions
print("\nActual vs Predicted Prices:")

for actual, predicted in zip(y_test.head(10), y_pred[:10]):
    print(
        f"Actual: ${actual:,.0f} | "
        f"Predicted: ${predicted:,.0f}"
    )

# Model coefficients
print("\nModel Coefficients:")
print("GrLivArea:", model.coef_[0])
print("BedroomAbvGr:", model.coef_[1])
print("FullBath:", model.coef_[2])

print("\nIntercept:", model.intercept_)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  : ${mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"R² Score: {r2 * 100:.2f}%")

import matplotlib.pyplot as plt

# Actual vs Predicted Prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
