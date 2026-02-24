from sklearn.linear_model import LinearRegression
import numpy as np

# Sample dataset
# Features: [hours studied, number of revisions]
X = np.array([
    [5, 2],
    [3, 1],
    [8, 3],
    [2, 0],
    [6, 2]
])

# Target: Test score
y = np.array([75, 50, 90, 40, 80])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict for new data
new_data = np.array([[7, 2]])  # 7 hours studied, 2 revisions
prediction = model.predict(new_data)
print("Predicted Score:", prediction[0])