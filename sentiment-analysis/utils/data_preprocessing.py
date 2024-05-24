import pandas as pd

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

# Reset index after removing rows
df_cleaned.reset_index(drop=True, inplace=True)

# Save the preprocessed data into the '../datasets' directory
df_cleaned.to_csv('../datasets/preprocessed_data.csv', index=False)