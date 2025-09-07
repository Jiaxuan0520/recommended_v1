import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# =====================================================================================
# == Functions for Streamlit App (main.py)
# =====================================================================================

def content_based_filtering_enhanced(movie_title, genre_input, df, top_n=10, genre_weight=0.3):
    """
    Generates movie recommendations based on content similarity for the Streamlit app.
    """
    df_copy = df.copy()
    df_copy.fillna('', inplace=True)
    df_copy['soup'] = df_copy['Genre'] + ' ' + df_copy['Overview'] + ' ' + df_copy['Director'] + ' ' + df_copy['Stars']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_copy.index, index=df_copy['Series_Title']).drop_duplicates()

    if movie_title not in indices:
        return pd.DataFrame()

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    if genre_input:
        genre_col = 'Genre_x' if 'Genre_x' in df_copy.columns else 'Genre'
        for i, score in sim_scores:
            if genre_input.lower() in str(df_copy.iloc[i][genre_col]).lower():
                sim_scores[i] = (i, score + genre_weight)

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_copy.iloc[movie_indices]

# =====================================================================================
# == Functions for Offline Evaluation (evaluate_recommendations.py)
# =====================================================================================

def create_content_features(df):
    """
    Pre-computes TF-IDF matrix, cosine similarity, and indices for evaluation.
    """
    df_copy = df.copy()
    df_copy.fillna('', inplace=True)
    df_copy['soup'] = df_copy['Genre'] + ' ' + df_copy['Overview'] + ' ' + df_copy['Director'] + ' ' + df_copy['Stars']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_copy.index, index=df_copy['Movie_ID']).drop_duplicates()
    
    return tfidf_matrix, cosine_sim, indices

def predict_content_ratings(user_id, train_df, movies_df, tfidf_matrix, cosine_sim, indices):
    """
    Predicts movie ratings for a user based on their past ratings for the evaluation script.
    Returns a dictionary of {movie_id: predicted_rating}.
    """
    user_ratings = train_df[train_df['User_ID'] == user_id]
    if user_ratings.empty:
        return {}

    # Get the indices of movies the user has rated
    rated_movie_indices = [indices[movie_id] for movie_id in user_ratings['Movie_ID'] if movie_id in indices]
    
    # Create a user profile: a weighted average of TF-IDF vectors of movies they rated
    user_profile = np.dot(user_ratings['Rating'].values, tfidf_matrix[rated_movie_indices])
    
    # FIX: Convert the resulting numpy.matrix to a standard numpy.ndarray
    user_profile = np.asarray(user_profile)
    
    # Reshape the user profile for the similarity calculation
    user_profile = user_profile.reshape(1, -1)

    # Calculate cosine similarity between the user profile and all movies
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    # Scale scores to a 1-10 rating
    scaler = MinMaxScaler(feature_range=(1, 10))
    scaled_scores = scaler.fit_transform(sim_scores.reshape(-1, 1)).flatten()
    
    # Create a dictionary of movie_id to predicted rating
    movie_ids = movies_df['Movie_ID'].iloc[indices.values]
    predictions = {movie_id: score for movie_id, score in zip(movie_ids, scaled_scores)}
    
    return predictions

