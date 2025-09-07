import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from content_based import content_based_filtering_enhanced, predict_content_ratings
from collaborative import collaborative_filtering_enhanced, train_svd_model, svd_model, predict_collaborative_ratings
import warnings

warnings.filterwarnings('ignore')

# =====================================================================================
# == Functions for Streamlit App (main.py)
# =====================================================================================

def smart_hybrid_recommendation(user_id, movie_title, genre_input, df, ratings_df, top_n=10, content_weight=0.5, collab_weight=0.5):
    """
    Generates hybrid recommendations for the Streamlit app.
    """
    # Ensure the collaborative model is trained (uses the cached version if available)
    global svd_model
    if svd_model is None:
        svd_model = train_svd_model(ratings_df)

    # 1. Get Content-Based Recommendations
    content_recs = content_based_filtering_enhanced(movie_title, genre_input, df, top_n=len(df), genre_weight=0.3)
    if content_recs.empty:
        return collaborative_filtering_enhanced(user_id, df, ratings_df, top_n=top_n)

    content_recs['content_score'] = range(len(content_recs), 0, -1)

    # 2. Get Collaborative Filtering Predictions
    user_rated_movies = ratings_df[ratings_df['User_ID'] == user_id]['Movie_ID'].tolist()
    unrated_movies = [movie_id for movie_id in df['Movie_ID'].unique() if movie_id not in user_rated_movies]
    
    predictions = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    collab_scores_df = pd.DataFrame([(pred.iid, pred.est) for pred in predictions], columns=['Movie_ID', 'collab_score'])

    # 3. Combine Scores
    hybrid_df = pd.merge(content_recs, collab_scores_df, on='Movie_ID', how='left')
    hybrid_df['collab_score'].fillna(0, inplace=True)

    scaler = MinMaxScaler()
    hybrid_df[['content_score', 'collab_score']] = scaler.fit_transform(hybrid_df[['content_score', 'collab_score']])
    hybrid_df['hybrid_score'] = (hybrid_df['content_score'] * content_weight) + (hybrid_df['collab_score'] * collab_weight)

    hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False)
    
    if movie_title in hybrid_df['Series_Title'].values:
        hybrid_df = hybrid_df[hybrid_df['Series_Title'] != movie_title]

    return hybrid_df.head(top_n)

# =====================================================================================
# == Functions for Offline Evaluation (evaluate_recommendations.py)
# =====================================================================================

def predict_hybrid_ratings(user_id, movie_id, train_df, movies_df, tfidf_matrix, cosine_sim, indices, svd_model, content_weight=0.5, collab_weight=0.5):
    """
    Predicts a single movie rating for a user by combining content and collaborative methods.
    """
    # 1. Get Content-Based Prediction
    # This is simplified for a single prediction to avoid re-calculating for all movies
    user_ratings = train_df[train_df['User_ID'] == user_id]
    content_pred_score = 3.0 # Default score
    if not user_ratings.empty and movie_id in indices:
        # Get indices of movies the user rated
        rated_movie_indices = [indices[mid] for mid in user_ratings['Movie_ID'] if mid in indices]
        
        # Get similarity of the target movie to the movies the user has rated
        target_movie_idx = indices[movie_id]
        sim_scores_to_rated = cosine_sim[target_movie_idx, rated_movie_indices]
        
        # Calculate a weighted average similarity based on user's ratings
        weighted_sim = np.dot(sim_scores_to_rated, user_ratings['Rating']) / user_ratings['Rating'].sum() if user_ratings['Rating'].sum() > 0 else 0
        
        # Scale the 0-1 similarity to a 1-10 rating
        content_pred_score = weighted_sim * 9 + 1

    # 2. Get Collaborative Prediction
    collab_pred_score = predict_collaborative_ratings(user_id, movie_id, svd_model)

    # 3. Combine with weights
    final_score = (content_pred_score * content_weight) + (collab_pred_score * collab_weight)
    
    return final_score

