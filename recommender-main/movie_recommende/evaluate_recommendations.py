# evaluate_recommendations.py (Optimized for better performance)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_error, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings

# Note: Run 'pip install scikit-surprise' locally to enable SVD evaluation
# This import is intentionally optional to avoid Streamlit deployment issues
try:
    from surprise import SVD, Reader, Dataset
    SURPRISE_AVAILABLE = True
    print("✅ Surprise library available - SVD evaluation enabled")
except ImportError:
    SURPRISE_AVAILABLE = False
    print("⚠️  Surprise library not available. Install with: pip install scikit-surprise")

warnings.filterwarnings('ignore')

# --- Optimized Configuration ---
RATING_THRESHOLD = 4.0  # Raised threshold - movies 6+ are "good"
TEST_SIZE = 0.15  # Smaller test set = more training data
RANDOM_STATE = 42
MIN_RATINGS_PER_USER = 5  # Filter sparse users
MIN_RATINGS_PER_MOVIE = 3  # Filter sparse movies

# --- Data Loading with Filtering ---
def load_and_prepare_data():
    movies = pd.read_csv('movies.csv')
    imdb = pd.read_csv('imdb_top_1000.csv')
    user_ratings = pd.read_csv('user_movie_rating.csv')

    if 'Movie_ID' not in movies.columns:
        movies['Movie_ID'] = range(len(movies))

    merged_df = pd.merge(movies, imdb, on='Series_Title', how='left')

    for col in ['Genre', 'Overview', 'Director', 'Stars']:
        col_x, col_y = f'{col}_x', f'{col}_y'
        if col_y in merged_df.columns:
            merged_df[col] = merged_df[col_y].fillna(merged_df.get(col_x, ''))
        elif col_x in merged_df.columns:
             merged_df[col] = merged_df[col_x]
        else:
             merged_df[col] = ''

    for col in ['IMDB_Rating', 'No_of_Votes', 'Released_Year']:
        if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    merged_df = merged_df.dropna(subset=['Movie_ID']).drop_duplicates(subset=['Movie_ID'])
    merged_df['Movie_ID'] = merged_df['Movie_ID'].astype(int)
    
    # OPTIMIZATION: Filter out sparse users and movies
    user_counts = user_ratings['User_ID'].value_counts()
    movie_counts = user_ratings['Movie_ID'].value_counts()
    
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    valid_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    
    user_ratings_filtered = user_ratings[
        (user_ratings['User_ID'].isin(valid_users)) & 
        (user_ratings['Movie_ID'].isin(valid_movies))
    ]
    
    print(f"Original ratings: {len(user_ratings)}")
    print(f"Filtered ratings: {len(user_ratings_filtered)} ({len(user_ratings_filtered)/len(user_ratings)*100:.1f}%)")
    print(f"Users: {user_ratings_filtered['User_ID'].nunique()}, Movies: {user_ratings_filtered['Movie_ID'].nunique()}")
    
    return merged_df, user_ratings_filtered

# --- Enhanced SVD Training ---
def train_svd_model(train_df):
    """Train optimized SVD model"""
    if not SURPRISE_AVAILABLE:
        return None
    
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(train_df[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset = data.build_full_trainset()
    
    # Optimized parameters for better performance
    svd = SVD(
        n_epochs=50,      # More training epochs
        n_factors=150,    # More latent factors
        lr_all=0.005,     # Lower learning rate for stability
        reg_all=0.02,     # Regularization to prevent overfitting
        random_state=RANDOM_STATE
    )
    svd.fit(trainset)
    return svd

def predict_svd_collaborative(user_id, movie_id, svd_model, global_mean):
    """Enhanced SVD prediction with confidence scoring"""
    if svd_model is None:
        return global_mean
    
    try:
        prediction = svd_model.predict(user_id, movie_id)
        # Use prediction confidence - if very uncertain, blend with global mean
        confidence = 1.0 - abs(prediction.est - global_mean) / 10.0
        confidence = max(0.1, confidence)  # Minimum confidence
        
        return prediction.est * confidence + global_mean * (1 - confidence)
    except Exception:
        return global_mean

# --- Enhanced Content-Based Prediction ---
def predict_content_based(user_id, movie_id, train_df, content_sim_matrix, movie_id_to_idx, global_mean):
    if movie_id not in movie_id_to_idx: 
        return global_mean
    
    user_ratings = train_df[train_df['User_ID'] == user_id]
    if user_ratings.empty: 
        return global_mean

    target_movie_idx = movie_id_to_idx[movie_id]
    sim_scores = content_sim_matrix[target_movie_idx]
    
    # Enhanced weighting: prefer high-similarity, high-rated movies
    numerator, denominator = 0, 0
    for _, row in user_ratings.iterrows():
        rated_movie_id = row['Movie_ID']
        if rated_movie_id in movie_id_to_idx:
            rated_movie_idx = movie_id_to_idx[rated_movie_id]
            similarity = sim_scores[rated_movie_idx]
            if similarity > 0.1:  # Only use reasonably similar movies
                # Weight by both similarity and rating deviation from mean
                rating_weight = 1.0 + abs(row['Rating'] - global_mean) / 10.0
                weight = similarity * rating_weight
                numerator += weight * row['Rating']
                denominator += weight
    
    if denominator > 0:
        predicted = numerator / denominator
        # Blend with global mean for stability
        return 0.8 * predicted + 0.2 * global_mean
    return global_mean

# --- Evaluation ---
if __name__ == "__main__":
    print("Loading and preparing data with optimization...")
    merged_df, user_ratings = load_and_prepare_data()
    
    # Stratified split to ensure rating distribution balance
    train_df, test_df = train_test_split(
        user_ratings, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=pd.cut(user_ratings['Rating'], bins=5, labels=False)  # Stratify by rating
    )
    global_mean_rating = train_df['Rating'].mean()
    print(f"Global mean rating: {global_mean_rating:.2f}")

    print("Building enhanced content-based similarity matrix...")
    # Enhanced feature weighting
    w_overview, w_genre, w_title, w_director, w_stars = 5, 4, 3, 3, 2
    soup = (
        (merged_df['Overview'].fillna('') + ' ') * w_overview +
        (merged_df['Genre'].fillna('') + ' ') * w_genre +
        (merged_df['Series_Title'].fillna('') + ' ') * w_title +
        (merged_df['Director'].fillna('') + ' ') * w_director +
        (merged_df['Stars'].fillna('') + ' ') * w_stars
    )

    # Optimized TF-IDF parameters
    tfidf = TfidfVectorizer(
        stop_words='english', 
        min_df=3, 
        max_df=0.7,
        ngram_range=(1, 2),  # Include bigrams
        max_features=10000   # Limit features for performance
    )
    tfidf_matrix = tfidf.fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    merged_df_reset = merged_df.reset_index(drop=True)
    movie_id_to_idx = {mid: i for i, mid in enumerate(merged_df_reset['Movie_ID'])}

    # Train SVD model
    if SURPRISE_AVAILABLE:
        print("Training optimized SVD collaborative filtering model...")
        svd_model = train_svd_model(train_df)
        print(f"SVD model trained on {len(train_df)} ratings")
    else:
        print("Skipping SVD model training (Surprise not available)")
        svd_model = None

    print("Evaluating models...")
    predictions = defaultdict(list)
    true_ratings = list(test_df['Rating'])

    for _, row in test_df.iterrows():
        user_id, movie_id = row['User_ID'], row['Movie_ID']
        
        # SVD Collaborative Filtering
        pred_svd = predict_svd_collaborative(user_id, movie_id, svd_model, global_mean_rating)
        predictions['collab'].append(pred_svd)
        
        # Enhanced Content-Based
        pred_cb = predict_content_based(user_id, movie_id, train_df, content_sim_matrix, movie_id_to_idx, global_mean_rating)
        predictions['content'].append(pred_cb)
        
        # Smart Hybrid with adaptive weighting
        user_rating_count = len(train_df[train_df['User_ID'] == user_id])
        if user_rating_count < 10:  # New users: favor content-based
            hybrid_pred = 0.3 * pred_svd + 0.7 * pred_cb
        else:  # Established users: favor collaborative
            hybrid_pred = 0.75 * pred_svd + 0.25 * pred_cb
        
        predictions['hybrid'].append(hybrid_pred)
    
    print(f"\n--- Evaluation Results ---\n")
    results = {}
    y_true_cls = [1 if r >= RATING_THRESHOLD else 0 for r in true_ratings]
    
    # Check class balance
    positive_ratio = np.mean(y_true_cls)
    print(f"Class distribution: {positive_ratio:.1%} positive, {1-positive_ratio:.1%} negative")
    
    for model_name in ['content', 'collab', 'hybrid']:
        y_pred_reg = np.clip(predictions[model_name], 1, 10)
        y_pred_cls = [1 if r >= RATING_THRESHOLD else 0 for r in y_pred_reg]
        
        results[model_name] = {
            'report': classification_report(y_true_cls, y_pred_cls, target_names=['negative', 'positive'], zero_division=0),
            'accuracy': accuracy_score(y_true_cls, y_pred_cls),
            'rmse': np.sqrt(mean_squared_error(true_ratings, y_pred_reg)),
            'precision': precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0),
            'mae': np.mean(np.abs(np.array(true_ratings) - np.array(y_pred_reg))),
            'f1': f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
        }
        
        title = model_name.replace('content', 'Content-Based').replace('collab', 'Collaborative').replace('hybrid', 'Hybrid')
        print(f"Model: {title}")
        print(f"Accuracy: {results[model_name]['accuracy']:.3f}")
        print(f"RMSE: {results[model_name]['rmse']:.3f}")
        print(f"MAE: {results[model_name]['mae']:.3f}")
        print(f"{results[model_name]['report']}")
        print(f"{'-'*60}")

    print("\n--- Performance Comparison ---\n")
    summary_df = pd.DataFrame({
        "Method": ["Content-Based", "Collaborative (SVD)", "Smart Hybrid"],
        "Accuracy": [res['accuracy'] for res in results.values()],
        "Precision": [res['precision'] for res in results.values()],
        "Recall": [res['recall'] for res in results.values()],
        "RMSE": [res['rmse'] for res in results.values()],
        "MAE": [res['mae'] for res in results.values()]
    })
    
    print(summary_df.to_string(
        index=False, 
        formatters={
            'Accuracy': '{:.3f}'.format,
            'Precision': '{:.3f}'.format, 
            'Recall': '{:.3f}'.format, 
            'RMSE': '{:.3f}'.format,
            'MAE': '{:.3f}'.format
        }
    ))
    
  