import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from models import nn_model

# Prepare the training and validation data
X_train_filename = '../datasets/X_train.csv'
y_train_filename = '../datasets/y_train.csv'
X_validation_filename = '../datasets/X_validation.csv'
y_validation_filename = '../datasets/y_validation.csv'

X_train_df = pd.read_csv(X_train_filename)
y_train_df = pd.read_csv(y_train_filename)
X_validation_df = pd.read_csv(X_validation_filename)
y_validation_df = pd.read_csv(y_validation_filename)

# Convert target variables to 1D arrays (required by scikit-learn)
y_train = y_train_df.values.ravel()
y_validation = y_validation_df.values.ravel()

# Converting inputs into numpy arrays
X_train = X_train_df.values
X_validation = X_validation_df.values

"""
Training the Linear Regression model
"""
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model weights
joblib.dump(model, 'linear_regression_model.pkl')

"""
Training the Neural Network model
"""
# # Train the model
# model = nn_model
# model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1)
#
# # Save the trained neural network model
# model.save('nn_model.h5')

"""
Standard testing function for the two models
"""

def validate_model(model, X_validation, y_validation):
    # Predict on the validation set
    y_pred = model.predict(X_validation)

    # Evaluate the model
    mse = mean_squared_error(y_validation, y_pred)
    r2 = r2_score(y_validation, y_pred)

    # Print the performance metrics
    print("Mean Squared Error (MSE) on validation set:", mse)
    print("R-squared (R2) on validation set:", r2)

validate_model(model, X_validation, y_validation)