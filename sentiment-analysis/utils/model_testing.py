import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from joblib import load

# Load the preprocessed data
test_df = pd.read_csv('../datasets/test_data.csv')

# Split the data into training and testing sets
X_test = test_df['reviews.text']
y_test = test_df['reviews.rating']

"""
Testing the Logistic Regression model
"""
# # Convert text data to TF-IDF features
# vectorizer = load('tfidf_vectorizer.joblib')
# X_test_tfidf = vectorizer.transform(X_test)
#
# # Load the trained Logistic Regression model
# model = load('logistic_regression_model.joblib')
#
# # Test the model
# y_pred = model.predict(X_test_tfidf)
# print("Classification Report:\n", classification_report(y_test, y_pred))


"""
Testing the RNN model
"""
# # Tokenize the text data
# tokenizer = load('tokenizer.joblib')
# X_test_seq = tokenizer.texts_to_sequences(X_test)
#
# # Pad the sequences to ensure uniform input size
# maxlen = 100
# X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
#
# # Load the trained RNN model
# model = load_model('rnn_model.keras')
#
# # Test the model
# y_pred = model.predict(X_test_pad)
# y_pred_binary = (y_pred > 0.5).astype(int)
# print("Test Classification Report:\n", classification_report(y_test, y_pred_binary))

"""
Testing the Naive Bayes model
"""
# # Load the trained Naive Bayes model
# model = load('naive_bayes_model.joblib')
#
# # Test the model
# y_pred = model.predict(X_test)
# print("Test Classification Report:\n", classification_report(y_test, y_pred))

"""
Testing the SVM
"""
# Load the trained SVM
model = load('svm_model.joblib')

# Test the model
y_pred = model.predict(X_test)
print("Test Classification Report:\n", classification_report(y_test, y_pred))