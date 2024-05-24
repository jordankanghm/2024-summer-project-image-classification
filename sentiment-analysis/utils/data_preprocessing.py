import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../datasets/amazon.csv')

# Remove rows where 'reviews.text' or 'reviews.rating' are missing
df_cleaned = df.dropna(subset=['reviews.text', 'reviews.rating'])

# Remove rows with 3 star ratings(neutral sentiment), and map 4-5 star reviews to 1(positive), and
# 1-2 star reviews to 0(negative).
df_cleaned = df_cleaned[df_cleaned['reviews.rating'] != 3]
df_cleaned['reviews.rating'] = df_cleaned['reviews.rating'].map({4: 1, 5: 1, 1: 0, 2: 0})

# Keep only the relevant columns
df_cleaned = df_cleaned[['reviews.text', 'reviews.rating']]

# Split the data into training(70%), validation(15%), and test sets(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Reset the indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the splits into separate CSV files
train_df.to_csv('../datasets/train_data.csv', index=False)
val_df.to_csv('../datasets/val_data.csv', index=False)
test_df.to_csv('../datasets/test_data.csv', index=False)

# Save the preprocessed data into the '../datasets' directory
df_cleaned.to_csv('../datasets/preprocessed_data.csv', index=False)