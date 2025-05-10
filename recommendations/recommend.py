import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Sample datasets
items = {
    'item_id': [1, 2, 3, 4],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'tags': [
        ['action', 'thriller'],
        ['comedy', 'drama'],
        ['action', 'adventure'],
        ['drama', 'romance']
    ]
}

ratings = {
    'user_id': [1, 1, 2, 2, 3],
    'item_id': [1, 2, 1, 3, 4],
    'rating': [5, 3, 4, 2, 4]
}

df_items = pd.DataFrame(items)
df_ratings = pd.DataFrame(ratings)

# Preprocess tags
mlb = MultiLabelBinarizer()
tag_matrix = mlb.fit_transform(df_items['tags'])
tag_df = pd.DataFrame(tag_matrix, columns=mlb.classes_, index=df_items['item_id'])

# Normalize ratings
scaler = MinMaxScaler()
df_ratings['rating'] = scaler.fit_transform(df_ratings[['rating']])

# Compute item-item similarity
tag_similarity = cosine_similarity(tag_df)
user_item_matrix = df_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
user_similarity = cosine_similarity(user_item_matrix)

# Combine similarities
def combine_similarity(tag_sim, user_sim, alpha=0.5):
    return alpha * tag_sim + (1 - alpha) * user_sim

combined_similarity = combine_similarity(tag_similarity, user_similarity)
combined_similarity_df = pd.DataFrame(combined_similarity, index=df_items['item_id'], columns=df_items['item_id'])

# Recommendation function
def recommend_items(user_id, df_ratings, combined_similarity_df, n=3):
    user_rated_items = df_ratings[df_ratings['user_id'] == user_id]['item_id'].values
    item_scores = combined_similarity_df[user_rated_items].mean(axis=0)

    item_scores = item_scores.drop(user_rated_items)
    recommended_items = item_scores.sort_values(ascending=False).head(n)
    return df_items[df_items['item_id'].isin(recommended_items.index)]

# Evaluate model
'''
def evaluate_recommendations(df_ratings, combined_similarity_df):
    actual_ratings = []
    predicted_ratings = []

    for _, row in df_ratings.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']

        predicted_items = recommend_items(user_id, df_ratings, combined_similarity_df, n=5)

        if item_id in predicted_items['item_id'].values:
            predicted_rating = predicted_items.loc[predicted_items['item_id'] == item_id].iloc[0]['rating']
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)

    mse = mean_squared_error(actual_ratings, predicted_ratings)
    return mse

# Test recommendations
mse = evaluate_recommendations(df_ratings, combined_similarity_df)
print(f'Mean Squared Error: {mse}')
'''

# Example recommendations for user 1
recommendations = recommend_items(1, df_ratings, combined_similarity_df)
print(recommendations)
