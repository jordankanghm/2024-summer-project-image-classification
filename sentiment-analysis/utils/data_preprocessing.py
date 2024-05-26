import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

"""
During data exploration, it was found that there was an extreme class imbalance between the positive
and negative ratings. Hence, class weights might not be suitable here as although the loss function
penalises misclassifications of the minority class, the overwhelming proportion of positive examples
could still result in an overall lower loss, causing the model to be trained inaccurately.

Random downsampling would also result in a large number of positive examples to be lost due to the
extreme class imbalance, resulting in the model being unable to learn to patterns encompassing the
positive class.

Random upsampling with regularisation is a viable option as all information about the data is retained
and overfitting is minimised by regularisation. However, I was afraid that there would be too few
examples for the model to train on. Hence, I decided to use SMOTE analysis as well to generate 
synthetic examples for the minority class. 
"""
# Load the dataset
df = pd.read_csv('../datasets/amazon.csv')

# Remove rows where 'reviews.text' or 'reviews.rating' are missing
df_cleaned = df.dropna(subset=['reviews.text', 'reviews.rating'])

# Remove rows with 3 star ratings (neutral sentiment), and map 4-5 star reviews to 1 (positive), and 1-2 star reviews to 0 (negative)
df_cleaned = df_cleaned[df_cleaned['reviews.rating'] != 3]
df_cleaned['reviews.rating'] = df_cleaned['reviews.rating'].map({4: 1, 5: 1, 1: 0, 2: 0})

"""
Random Upsampling
"""
# # Separate majority and minority classes
# df_majority = df_cleaned[df_cleaned['reviews.rating'] == 1]
# df_minority = df_cleaned[df_cleaned['reviews.rating'] == 0]
#
# # Upsample minority class
# df_minority_upsampled = resample(df_minority,
#                                  replace=True,     # Sample with replacement
#                                  n_samples=len(df_majority),    # Match majority class size
#                                  random_state=42)  # Reproducible results
#
# # Combine majority class with upsampled minority class
# df_final = pd.concat([df_majority, df_minority_upsampled])

"""
SMOTE Analysis
"""
# Split the data into features and target
X = df_cleaned['reviews.text']
y = df_cleaned['reviews.rating']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_smote_tfidf, y_smote = smote.fit_resample(X_tfidf, y)

# Convert the resampled TF-IDF features back to text
inverse_transform = vectorizer.inverse_transform(X_smote_tfidf)
reconstructed_text = [' '.join(doc) for doc in inverse_transform]

# Filter out empty strings and corresponding labels
filtered_indices = [i for i, text in enumerate(reconstructed_text) if text.strip()]
reconstructed_text = [reconstructed_text[i] for i in filtered_indices]
y_smote_filtered = [y_smote[i] for i in filtered_indices]

# Create the DataFrame for the resampled data
df_final = pd.DataFrame({'reviews.text': reconstructed_text, 'reviews.rating': y_smote_filtered})

"""
Split the datasets into training, validation and test sets
"""
# Split the data into training (70%), validation (15%), and test sets (15%)
train_df, temp_df = train_test_split(df_final, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Reset the indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the splits into separate CSV files
train_df.to_csv('../datasets/train_data.csv', index=False)
val_df.to_csv('../datasets/val_data.csv', index=False)
test_df.to_csv('../datasets/test_data.csv', index=False)
