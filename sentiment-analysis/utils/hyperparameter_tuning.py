import keras_tuner as kt
import pandas as pd
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load the preprocessed data
train_df = pd.read_csv('../datasets/train_data.csv')
val_df = pd.read_csv('../datasets/val_data.csv')

# Split the data into training and validation sets
X_train = train_df['reviews.text']
y_train = train_df['reviews.rating']
X_val = val_df['reviews.text']
y_val = val_df['reviews.rating']

# Model architecture
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=hp.Int('embedding_dim', min_value=32, max_value=256, step=32)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('lstm_units', min_value=32, max_value=256, step=32),
                   dropout=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1),
                   recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # Try 20 different hyperparameter combinations
    executions_per_trial=3, # Validate each combination 3 times to account for variance
    directory='.',
    project_name='hyperparameter_tuning_results'
)

# Tokenize and pad input data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

maxlen = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)

# Start the hyperparameter search
tuner.search(X_train_pad, y_train, epochs=5, validation_data=(X_val_pad, y_val), batch_size=32)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal embedding dimension is {best_hps.get('embedding_dim')},
the optimal number of LSTM units is {best_hps.get('lstm_units')},
the optimal dropout rates are {best_hps.get('dropout_1')} and {best_hps.get('dropout_2')},
the optimal recurrent dropout rate is {best_hps.get('recurrent_dropout')},
and the optimal learning rate is {best_hps.get('lr')}.
""")

# Build the best model
model = tuner.hypermodel.build(best_hps)

# Train the best model
model.fit(X_train_pad, y_train, epochs=5, validation_data=(X_val_pad, y_val), batch_size=32)

# Save the best model
model.save('best_rnn_model.keras')

# Validate the best model
y_pred = model.predict(X_val_pad)
y_pred_binary = (y_pred > 0.5).astype(int)
print("Test Classification Report:\n", classification_report(y_val, y_pred_binary))
