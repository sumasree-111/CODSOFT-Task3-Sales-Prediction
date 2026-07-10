import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("advertising.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
plt.figure(figsize=(6,4))
sns.histplot(df["Sales"], bins=15, kde=True)
plt.title("Sales Distribution")
plt.savefig("images/sales_distribution.png")
plt.show()
# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model Trained Successfully!")
# Predict
y_pred = model.predict(X_test)

print("First 5 Predictions:")
print(y_pred[:5])
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation")
print("R2 Score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("images/actual_vs_predicted_sales.png")
plt.show()
# Sample Prediction
sample = [[230.1, 37.8, 69.2]]

prediction = model.predict(sample)

print("\nPredicted Sales:", prediction[0])