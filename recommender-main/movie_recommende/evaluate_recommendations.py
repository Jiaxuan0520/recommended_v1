# evaluate_recommendations.py (Aligned with requirements: weighted TF-IDF, item-KNN, hybrid blend)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_error, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Configuration ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_RATINGS_PER_USER = 3
MIN_RATINGS_PER_MOVIE = 3


def load_and_prepare_data():
    movies = pd.read_csv('movies.csv')
    imdb = pd.read_csv('imdb_top_1000.csv')
    ratings = pd.read_csv('user_movie_rating.csv')

    if 'Movie_ID' not in movies.columns:
        movies['Movie_ID'] = range(len(movies))

    merged = pd.merge(movies, imdb, on='Series_Title', how='left')

    # Normalize key columns (compute medians on numeric-converted series)
    if 'IMDB_Rating' in merged.columns:
        _imdb_num = pd.to_numeric(merged['IMDB_Rating'], errors='coerce')
        _imdb_med = float(np.nanmedian(_imdb_num)) if np.isfinite(np.nanmedian(_imdb_num)) else 0.0
        merged['IMDB_Rating'] = _imdb_num.fillna(_imdb_med)
    if 'No_of_Votes' in merged.columns:
        merged['No_of_Votes'] = pd.to_numeric(merged['No_of_Votes'], errors='coerce').fillna(0)
    if 'Released_Year' in merged.columns:
        _yr_num = pd.to_numeric(merged['Released_Year'], errors='coerce')
        _yr_med = float(np.nanmedian(_yr_num)) if np.isfinite(np.nanmedian(_yr_num)) else 2000.0
        merged['Released_Year'] = _yr_num.fillna(_yr_med)

    # Ensure consistent Genre column
    if 'Genre_y' in merged.columns:
        merged['Genre'] = merged['Genre_y'].fillna(merged.get('Genre_x', ''))
    elif 'Genre_x' in merged.columns:
        merged['Genre'] = merged['Genre_x']
    else:
        merged['Genre'] = merged.get('Genre', '')

    merged = merged.drop_duplicates(subset=['Movie_ID']).dropna(subset=['Movie_ID'])
    merged['Movie_ID'] = merged['Movie_ID'].astype(int)

    # Filter sparse users/movies for stability
    user_counts = ratings['User_ID'].value_counts()
    movie_counts = ratings['Movie_ID'].value_counts()
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    valid_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    ratings_f = ratings[(ratings['User_ID'].isin(valid_users)) & (ratings['Movie_ID'].isin(valid_movies))]

    print(f"Loaded ratings: {len(ratings)} → filtered: {len(ratings_f)}")
    return merged, ratings_f


def build_weighted_content_features(merged_df: pd.DataFrame):
    """Build weighted TF-IDF feature matrix per requirements.
    Weights: Genre=8, Rating=1.5, Title=0.5
    Returns: (feature_matrix, id_to_index, index_to_id)
    """
    df = merged_df.copy()
    genre_col = 'Genre'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else ('Rating' if 'Rating' in df.columns else None)

    df['Series_Title'] = df['Series_Title'].astype(str)
    df[genre_col] = df[genre_col].fillna('')

    tfidf_genre = TfidfVectorizer(stop_words='english')
    tfidf_title = TfidfVectorizer(stop_words='english')

    m_genre = tfidf_genre.fit_transform(df[genre_col].astype(str))
    m_title = tfidf_title.fit_transform(df['Series_Title'].astype(str))

    if rating_col:
        rating_vals = df[rating_col].fillna(df[rating_col].median()).to_numpy().reshape(-1, 1)
        rating_scaler = MinMaxScaler()
        rating_scaled = rating_scaler.fit_transform(rating_vals)
        m_rating = csr_matrix(rating_scaled)
    else:
        m_rating = csr_matrix(np.zeros((len(df), 1)))

    feature_matrix = hstack([
        m_genre.multiply(8.0),
        m_title.multiply(0.5),
        m_rating.multiply(1.5)
    ]).tocsr()

    id_to_index = pd.Series(df.index.values, index=df['Movie_ID'].astype(int)).to_dict()
    index_to_id = pd.Series(df['Movie_ID'].astype(int).values, index=df.index).to_dict()
    return feature_matrix, id_to_index, index_to_id


def build_item_item_similarity_from_ratings(ratings_df: pd.DataFrame, all_movie_ids: np.ndarray):
    pivot = ratings_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    item_vectors = pivot.reindex(columns=all_movie_ids, fill_value=np.nan)
    item_vectors = item_vectors.fillna(0.0).to_numpy().T  # shape: items x users
    # cosine similarity
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True) + 1e-8
    normalized = item_vectors / norms
    sim = normalized @ normalized.T
    return sim, {mid: i for i, mid in enumerate(all_movie_ids)}


def predict_rating_content(user_id, movie_id, train_df, content_sim, id_to_index, global_mean, top_k=30):
    if movie_id not in id_to_index:
        return global_mean
    user_hist = train_df[train_df['User_ID'] == user_id]
    if user_hist.empty:
        return global_mean
    target_idx = id_to_index[movie_id]
    sims = content_sim[target_idx]
    pairs = []
    for _, row in user_hist.iterrows():
        mid = int(row['Movie_ID'])
        if mid in id_to_index:
            sim = sims[id_to_index[mid]]
            if sim > 0:
                pairs.append((sim, row['Rating']))
    if not pairs:
        return global_mean
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:top_k]
    sims_arr = np.array([p[0] for p in pairs])
    ratings_arr = np.array([p[1] for p in pairs])
    pred = (sims_arr @ ratings_arr) / (sims_arr.sum() + 1e-8)
    # Blend for stability
    return 0.8 * pred + 0.2 * global_mean


def predict_rating_cf_itemknn(user_id, movie_id, train_df, item_sim, movie_id_to_pos, global_mean, top_k=30):
    if movie_id not in movie_id_to_pos:
        return global_mean
    user_hist = train_df[train_df['User_ID'] == user_id]
    if user_hist.empty:
        return global_mean
    target_pos = movie_id_to_pos[movie_id]
    pairs = []
    for _, row in user_hist.iterrows():
        mid = int(row['Movie_ID'])
        if mid in movie_id_to_pos:
            sim = item_sim[target_pos, movie_id_to_pos[mid]]
            if sim > 0:
                pairs.append((sim, row['Rating']))
    if not pairs:
        return global_mean
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:top_k]
    sims_arr = np.array([p[0] for p in pairs])
    ratings_arr = np.array([p[1] for p in pairs])
    pred = (sims_arr @ ratings_arr) / (sims_arr.sum() + 1e-8)
    return 0.85 * pred + 0.15 * global_mean


def compute_popularity_recency_scores(merged_df: pd.DataFrame):
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    votes_col = 'No_of_Votes' if 'No_of_Votes' in merged_df.columns else None
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else ('Year' if 'Year' in merged_df.columns else None)

    if rating_col and votes_col and votes_col in merged_df.columns:
        ratings = pd.to_numeric(merged_df[rating_col], errors='coerce').fillna(merged_df[rating_col].median())
        votes = pd.to_numeric(merged_df[votes_col], errors='coerce').fillna(0.0)
        popularity = ratings * np.log1p(votes)
    else:
        popularity = pd.Series(np.zeros(len(merged_df)))

    if year_col and year_col in merged_df.columns:
        years = pd.to_numeric(merged_df[year_col], errors='coerce').fillna(merged_df[year_col].median())
        current_year = datetime.now().year
        k = 15.0
        recency = np.exp(-(current_year - years) / k)
    else:
        recency = pd.Series(np.zeros(len(merged_df)))

    # Scale to rating range 1..10
    scaler = MinMaxScaler(feature_range=(1.0, 10.0))
    pop_scaled = scaler.fit_transform(popularity.to_numpy().reshape(-1, 1)).flatten()
    rec_scaled = scaler.fit_transform(recency.to_numpy().reshape(-1, 1)).flatten()
    return pop_scaled, rec_scaled


if __name__ == "__main__":
    print("Loading and preparing data...")
    merged_df, ratings_df = load_and_prepare_data()

    # Split ratings
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=pd.cut(ratings_df['Rating'], bins=5, labels=False)
    )
    global_mean = train_df['Rating'].mean()
    rating_threshold = float(np.median(train_df['Rating']))  # Balanced classes
    print(f"Global mean rating: {global_mean:.2f} | Classification threshold: {rating_threshold:.2f}")

    # Build content features and similarities
    feature_matrix, id_to_index, index_to_id = build_weighted_content_features(merged_df)
    content_sim = cosine_similarity(feature_matrix, feature_matrix)

    # Build item-item similarity from ratings (item-based KNN)
    all_movie_ids = merged_df['Movie_ID'].astype(int).to_numpy()
    item_sim, movie_id_to_pos = build_item_item_similarity_from_ratings(train_df, all_movie_ids)

    # Popularity and Recency
    pop_scaled, rec_scaled = compute_popularity_recency_scores(merged_df)
    pop_by_movie = {int(mid): pop for mid, pop in zip(merged_df['Movie_ID'].astype(int), pop_scaled)}
    rec_by_movie = {int(mid): rec for mid, rec in zip(merged_df['Movie_ID'].astype(int), rec_scaled)}

    # Predictions
    preds_content, preds_cf, preds_hybrid, y_true = [], [], [], []
    for _, row in test_df.iterrows():
        uid = row['User_ID']
        mid = int(row['Movie_ID'])
        true_rating = float(row['Rating'])
        y_true.append(true_rating)

        p_cb = predict_rating_content(uid, mid, train_df, content_sim, id_to_index, global_mean)
        p_cf = predict_rating_cf_itemknn(uid, mid, train_df, item_sim, movie_id_to_pos, global_mean)

        # Hybrid: 0.4 * content + 0.4 * CF + 0.1 * popularity + 0.1 * recency
        p_pop = pop_by_movie.get(mid, 5.0)
        p_rec = rec_by_movie.get(mid, 5.0)
        p_hb = 0.4 * p_cb + 0.4 * p_cf + 0.1 * p_pop + 0.1 * p_rec

        preds_content.append(np.clip(p_cb, 1, 10))
        preds_cf.append(np.clip(p_cf, 1, 10))
        preds_hybrid.append(np.clip(p_hb, 1, 10))

    # Helper to compute metrics and print report
    def evaluate_and_print(name, y_pred):
        y_pred = np.array(y_pred)
        y_true_arr = np.array(y_true)
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
        mse = float(mean_squared_error(y_true_arr, y_pred))
        y_true_cls = (y_true_arr >= rating_threshold).astype(int)
        y_pred_cls = (y_pred >= rating_threshold).astype(int)

        acc = float(accuracy_score(y_true_cls, y_pred_cls))
        prec = float(precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0))
        rec = float(recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0))
        f1 = float(f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0))
        report = classification_report(y_true_cls, y_pred_cls, target_names=['negative', 'positive'], zero_division=0)

        print(f"Model: {name}")
        print(f"Accuracy: {acc:.3f}")
        print(report)
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print("-" * 60)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'rmse': rmse,
            'mse': mse
        }

    print("\nEvaluating models...\n")
    res_content = evaluate_and_print('Content-Based', preds_content)
    res_collab = evaluate_and_print('Collaborative', preds_cf)
    res_hybrid = evaluate_and_print('Hybrid', preds_hybrid)

    print("Comparison table:")
    comp_df = pd.DataFrame([
        ['Collaborative', f"{res_collab['precision']:.2f}", f"{res_collab['recall']:.2f}", f"{res_collab['rmse']:.2f}", 'Worked well with dense ratings'],
        ['Content-Based', f"{res_content['precision']:.2f}", f"{res_content['recall']:.2f}", f"{res_content['rmse']:.2f}", 'Good with rich metadata'],
        ['Hybrid', f"{res_hybrid['precision']:.2f}", f"{res_hybrid['recall']:.2f}", f"{res_hybrid['rmse']:.2f}", 'Best balance between both']
    ], columns=['Method Used', 'Precision', 'Recall', 'RMSE', 'Notes'])
    print(comp_df.to_string(index=False))
