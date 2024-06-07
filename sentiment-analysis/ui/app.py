from flask import Flask, request, render_template
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__)

# Load the tokenizer
tokenizer_path = os.path.join('../utils', 'tokenizer.joblib')
tokenizer = joblib.load(tokenizer_path)

# Load the model
model_path = os.path.join('../utils', 'best_rnn_model.keras')
model = tf.keras.models.load_model(model_path)

def preprocess_review(review, tokenizer, maxlen=100):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def predict_sentiment(review):
    input_data = preprocess_review(review, tokenizer)

    with tf.device('/CPU:0'):
        logits = model.predict(input_data)
    predicted_class = (logits > 0.5).astype(int)[0][0]
    return 'Positive' if predicted_class == 1 else 'Negative'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('result.html', review=review, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
