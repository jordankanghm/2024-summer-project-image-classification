from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

"""
The models will be trained using a randomly upsampled dataset, as well as a SMOTE dataset. The test
statistics generated will be compared to identify the better dataset for each model.
"""

"""
As the task involves assigning texts to positive(1) and negative(0) classes, I decided to try using a 
logistic regression model. The text is converted into TF-IDF format. 

TF(Term Frequency) assigns greater importance to words corresponding to its frequency in a document. 
Inverse Document Frequency(IDF) assigns lesser importance to words with high frequency across documents
as they are likely to be less important.

The following shows the test scores for the logistic regression model.

Classification Report:
SMOTE DATASET
               precision    recall  f1-score   support

           0       0.97      0.99      0.98      4831
           1       0.99      0.97      0.98      4862

    accuracy                           0.98      9693
   macro avg       0.98      0.98      0.98      9693
weighted avg       0.98      0.98      0.98      9693

RANDOM UPSAMPLING DATASET
               precision    recall  f1-score   support

           0       0.95      0.99      0.97      4884
           1       0.99      0.95      0.97      4811

    accuracy                           0.97      9695
   macro avg       0.97      0.97      0.97      9695
weighted avg       0.97      0.97      0.97      9695

It can be seen that the model generally performs better on the SMOTE dataset.
"""

"""
I also tried to train a Recurrent Neural Network model. Since the data is sequential, an RNN would
perform well on the data.

The following shows the test scores for the RNN model.

Test Classification Report:
SMOTE DATASET
               precision    recall  f1-score   support

           0       1.00      0.98      0.99      4831
           1       0.98      1.00      0.99      4862

    accuracy                           0.99      9693
   macro avg       0.99      0.99      0.99      9693
weighted avg       0.99      0.99      0.99      9693

RANDOM UPSAMPLING DATASET
               precision    recall  f1-score   support

           0       0.99      1.00      0.99      4884
           1       1.00      0.99      0.99      4811

    accuracy                           0.99      9695
   macro avg       0.99      0.99      0.99      9695
weighted avg       0.99      0.99      0.99      9695

It can be seen that the model generally performs better on the random upsampling dataset.
"""
rnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    Dropout(0.5),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

"""
I also tried training a Naive Bayes model. Naive Bayes works based on the assumption that given a label,
the words in a review are independent of each other. Although this may not hold true in a review(e.g.
phrases like "battery" and "life" are likely correlated), it still performs well for text classification.

The following shows the test scores for the Naive Bayes model.

Classification Report:
SMOTE DATASET
               precision    recall  f1-score   support

           0       0.93      0.97      0.95      4831
           1       0.97      0.92      0.94      4862

    accuracy                           0.95      9693
   macro avg       0.95      0.95      0.95      9693
weighted avg       0.95      0.95      0.95      9693

RANDOM UPSAMPLING DATASET
               precision    recall  f1-score   support

           0       0.92      0.96      0.94      4884
           1       0.96      0.92      0.94      4811

    accuracy                           0.94      9695
   macro avg       0.94      0.94      0.94      9695
weighted avg       0.94      0.94      0.94      9695

It can be seen that the model generally performs better on the SMOTE dataset.
"""

"""
Support Vector Machines can also be used to classify the reviews. It works by finding the decision
boundary which maximises the margin between the positive and negative examples.

The following shows the test scores for the SVM.

Classification Report:
SMOTE DATASET
               precision    recall  f1-score   support

           0       0.98      1.00      0.99      4831
           1       1.00      0.98      0.99      4862

    accuracy                           0.99      9693
   macro avg       0.99      0.99      0.99      9693
weighted avg       0.99      0.99      0.99      9693

RANDOM UPSAMPLING DATASET
               precision    recall  f1-score   support

           0       0.97      1.00      0.98      4884
           1       1.00      0.97      0.98      4811

    accuracy                           0.98      9695
   macro avg       0.98      0.98      0.98      9695
weighted avg       0.98      0.98      0.98      9695

It can be seen that the model generally performs better on the SMOTE dataset.
"""

"""
Considering all possible models and datasets, an RNN model trained on a randomly upsampled dataset
produces the best results. Hence, we will perform hyperparameter tuning on this model.
"""