import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

csv_filename = '../datasets/stock_data.csv'
df = pd.read_csv(csv_filename)

# Drop 'Date' and 'Volume' features as they have low correlation with the target variable
df.drop(columns=['Date', 'Volume'], inplace=True)

# Splitting data into features and target variable
X = df.drop(columns=['Adj Close'])
y = df['Adj Close']

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
# Split the data into training and temporary sets (80% training, 20% temporary)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Further split the temporary data into validation and test sets (50% validation, 50% test)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert NumPy arrays back to DataFrames
X_train_temp_df = pd.DataFrame(X_train_temp, columns=X.columns)
y_train_temp_df = pd.DataFrame(y_train_temp, columns=['Adj Close'])
X_validation_df = pd.DataFrame(X_validation, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
y_validation_df = pd.DataFrame(y_validation, columns=['Adj Close'])
y_test_df = pd.DataFrame(y_test, columns=['Adj Close'])

# Save the datasets to CSV files
X_train_temp_df.to_csv('../datasets/X_train.csv', index=False)
X_validation_df.to_csv('../datasets/X_validation.csv', index=False)
X_test_df.to_csv('../datasets/X_test.csv', index=False)
y_train_temp_df.to_csv('../datasets/y_train.csv', index=False, header=True)
y_validation_df.to_csv('../datasets/y_validation.csv', index=False, header=True)
y_test_df.to_csv('../datasets/y_test.csv', index=False, header=True)
