# 2024-Summer-Machine-Learning-Project
This repository stores all machine learning projects I have worked on during the summer of 2024.
I have done three projects: Image Classification, Stock Price Prediction, and Sentiment Analysis.

## Start up
1. **Clone the repository**: Pull the remote repository into your local repository.
    ```bash
    git clone https://github.com/jordankanghm/2024-summer-projects-1.git
    ```

2. **Install dependencies**: Navigate to the project directory and install all dependencies listed in the requirements file.
    ```bash
    pip install -r requirements
    ```

3. **Navigate to the UI directory**: Go to the `ui` directory of the image-classification and sentiment-analysis directories.
    ```bash
    cd image-classification/ui
    cd sentiment-analysis/ui
    ```

5. **Run the application**: Execute the following command to start the application.
    ```bash
    python app.py
    ```

6. **Open your browser**: Go to the following URL to access the application.
    ```
    http://127.0.0.1:5000/
    ```

7. **Test the model**: Feel free to try out any inputs to test the model.

## Image Classification
I decided to conduct image classification on the MNIST dataset comprising handwritten digits. The dataset consists of 60,000 training and 10,000 testing images and labels. The image-classification directory shows all the work I have done, including data preprocessing, training and testing models, and hyperparameter tuning. The final model produced the following test scores.

Accuracy: 0.9907
Precision: 0.9909
Recall: 0.9905
F1: 0.9907

## Stock Price Prediction
I decided to conduct stock price prediction using Apple's stock price data in 2024. The dataset consists of 250 data points, consisting of features such as high, low and closing price. The stock-price-prediction directory shows all the work I have done, including data preprocessing, and training and testing models. The final model produced the following test scores.

Mean Squared Error: 0.02743
R-squared: 0.9999

## Amazon Product Review Sentiment Analysis
I decided to conduct sentiment analysis on user reviews for Amazon products. The dataset consists of 34,660 reviews, with reviews ranging from 1 star to 5 stars. 1 and 2 star reviews were treated as negative reivews and 4 and 5 star reviews were treated as positive reviews. The model was trained to predict whether a given review has a positive or negative connotation. The sentiment-analysis directory shows all the work I have done, including data preprocessing, training and testing models, and hyperparameter tuning. The final model produced the following test scores. Note that '0' represents the negative class and '1' represents the positive class.

               precision    recall  f1-score   support

           0       0.99      0.99      0.99      4831
           1       0.99      0.99      0.99      4862

    accuracy                           0.99      9693
   macro avg       0.99      0.99      0.99      9693
weighted avg       0.99      0.99      0.99      9693
