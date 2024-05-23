from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

"""
In order to predict adjusted close price, I used a Linear Regression model and a Neural Network model.
Each model only used "Open", "High", "Low" and "Close" as the training features, and "Adj Close" as the
target variable. "Date" and "Volume" were omitted as they were found to have low correlation with the
target variable, as can be seen in notebooks/data-exploration.ipynb.
"""


"""
The Linear Regression model produced a low mean squared error and a high R-squared statistic.
When tested against the test set, the average MSE was 0.02743 and R-squared statistic was 0.9999.
"""

"""
Below shows the neural network model used. The neural network model produced a low mean squared error
and a high R-squared statistic, though not as low as the linear regression model. The training time was
also longer compared to the linear regression model. When tested against the test set, the average MSE 
was 4.816 and R-squared statistic was 0.9846.
"""
# Define the neural network model
nn_model = Sequential()
nn_model.add(Dense(128, input_dim=4, activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation='linear'))

# Compile the model with a custom optimizer and learning rate
optimizer = Adam(learning_rate=0.001)
nn_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])