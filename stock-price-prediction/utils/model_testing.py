import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model

# Load the test data
X_test_filename = '../datasets/X_test.csv'
y_test_filename = '../datasets/y_test.csv'

X_test_df = pd.read_csv(X_test_filename)
y_test_df = pd.read_csv(y_test_filename)

X_test = X_test_df.values
y_test = y_test_df.values.ravel()

# Load the saved linear regression model
model = joblib.load('linear_regression_model.pkl')

# # Load the saved neural network model
# model = load_model('nn_model.h5')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R2) scores
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Mean Squared Error (MSE) on test set:", mse)
print("R-squared (R2) on test set:", r2)
