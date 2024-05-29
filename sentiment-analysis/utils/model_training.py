import numpy as np
import pandas as pd
import models
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import dump, load

# Load the preprocessed data
train_df = pd.read_csv('../datasets/train_data.csv')
val_df = pd.read_csv('../datasets/val_data.csv')

# Split the data into training and validation sets
X_train = train_df['reviews.text']
y_train = train_df['reviews.rating']
X_val = val_df['reviews.text']
y_val = val_df['reviews.rating']

"""
Train the Logistic Regression model
"""
# # Convert text data to TF-IDF features
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_val_tfidf = vectorizer.transform(X_val)
#
# # Train a Logistic Regression model(with L2 regularisation)
# model = LogisticRegression(penalty='l2', C=1.0)
# model.fit(X_train_tfidf, y_train)
#
# # Save the model and vectorizer
# dump(model, 'logistic_regression_model.joblib')
# dump(vectorizer, 'tfidf_vectorizer.joblib')
#
# # Validate the model
# y_pred = model.predict(X_val_tfidf)
# print("Accuracy:", accuracy_score(y_val, y_pred))
# print("Classification Report:\n", classification_report(y_val, y_pred))

"""
Train the RNN model
"""
# # Tokenize the text data
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(X_train)
#
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_val_seq = tokenizer.texts_to_sequences(X_val)
#
# # Pad the sequences to ensure uniform input size
# maxlen = 100
# X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
# X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)
#
# # Compile the model
# model = models.rnn_model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_val_pad, y_val))
#
# # Save the model and tokenizer
# model.save('rnn_model.keras')
# dump(tokenizer, 'tokenizer.joblib')
#
# # Validate the model
# y_pred = model.predict(X_val_pad)
# y_pred_binary = (y_pred > 0.5).astype(int)
# print("Test Classification Report:\n", classification_report(y_val, y_pred_binary))

"""
Train the Naive Bayes model
"""
# # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Save the trained model
# dump(model, 'naive_bayes_model.joblib')
#
# # Validate the model
# y_pred = model.predict(X_val)
# print("Test Classification Report:\n", classification_report(y_val, y_pred))

"""
Train the SVM
"""
# # Create a pipeline with TF-IDF vectorizer and SVM classifier
# model = make_pipeline(TfidfVectorizer(max_features=5000), LinearSVC())
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Save the trained model
# dump(model, 'svm_model.joblib')
#
# # Validate the model
# y_pred = model.predict(X_val)
# print("Test Classification Report:\n", classification_report(y_val, y_pred))