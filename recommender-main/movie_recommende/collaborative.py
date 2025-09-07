import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors


@st.cache_data
def load_user_ratings():
    # Prefer session state
    try:
        if 'user_ratings_df' in st.session_state:
            df = st.session_state['user_ratings_df']
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    # Fallback to local CSV
    try:
        return pd.read_csv('user_movie_rating.csv')
    except Exception:
        return None


def _build_user_item_matrix(ratings_df: pd.DataFrame, movie_ids: np.ndarray) -> pd.DataFrame:
    if ratings_df is None or ratings_df.empty:
        return None
    df = ratings_df[ratings_df['Movie_ID'].isin(movie_ids)].copy()
    if df.empty:
        return None
    return df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')


def _fit_item_knn(user_item: pd.DataFrame):
    if user_item is None or user_item.empty:
        return None, None
    item_vectors = user_item.fillna(0.0).T
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(item_vectors)
    return model, item_vectors


def _nearest_items(model, item_vectors, target_movie_id: int, k: int) -> dict:
    if model is None or item_vectors is None or target_movie_id not in item_vectors.index:
        return {}
    idx = item_vectors.index.get_loc(target_movie_id)
    distances, indices = model.kneighbors(item_vectors.iloc[[idx]], n_neighbors=min(k + 1, len(item_vectors)))
    neighbors = {}
    for d, i in zip(distances[0], indices[0]):
        nb_movie = int(item_vectors.index[i])
        if nb_movie == target_movie_id:
            continue
        neighbors[nb_movie] = 1.0 - float(d)
    return neighbors


@st.cache_data
def collaborative_knn(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k_neighbors: int = 20):
    if not isinstance(target_movie, str) or not target_movie.strip():
        return None
    if 'Movie_ID' not in merged_df.columns or 'Series_Title' not in merged_df.columns:
        return None

    # Map title -> Movie_ID (case-insensitive)
    titles_lower = merged_df['Series_Title'].astype(str).str.lower()
    mask = titles_lower == target_movie.strip().lower()
    if not mask.any():
        return None
    target_movie_id = int(merged_df.loc[mask, 'Movie_ID'].iloc[0])

    ratings_df = load_user_ratings()
    user_item = _build_user_item_matrix(ratings_df, merged_df['Movie_ID'].values)
    model, item_vectors = _fit_item_knn(user_item)
    neighbors = _nearest_items(model, item_vectors, target_movie_id, k=k_neighbors)
    if not neighbors:
        return None

    # Sort by similarity and take top N
    sorted_pairs = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [mid for mid, _ in sorted_pairs]
    sim_map = dict(sorted_pairs)

    # Prepare result
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    cols = ['Series_Title', 'Movie_ID'] + ([genre_col] if genre_col else []) + ([rating_col] if rating_col else [])
    subset = merged_df[merged_df['Movie_ID'].isin(top_ids)][cols].drop_duplicates(['Series_Title','Movie_ID'])

    # Preserve order
    id_to_rank = {mid: i for i, mid in enumerate(top_ids)}
    subset = subset.copy()
    subset['rank_order'] = subset['Movie_ID'].map(id_to_rank)
    subset['Similarity'] = subset['Movie_ID'].map(sim_map)
    subset = subset.sort_values('rank_order').drop(columns=['rank_order', 'Movie_ID'])
    return subset


@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8):
    return collaborative_knn(merged_df, target_movie, top_n=top_n)


@st.cache_data
def diagnose_data_linking(merged_df: pd.DataFrame):
    issues = {
        'has_movie_id': 'Movie_ID' in merged_df.columns,
        'unique_titles': merged_df['Series_Title'].nunique(),
        'rows': len(merged_df)
    }
    try:
        ratings = load_user_ratings()
        issues['ratings_loaded'] = ratings is not None and not ratings.empty
        if issues['ratings_loaded'] and issues['has_movie_id']:
            issues['ratings_coverage_ratio'] = float(ratings['Movie_ID'].isin(merged_df['Movie_ID']).mean())
    except Exception:
        issues['ratings_loaded'] = False
    return issues