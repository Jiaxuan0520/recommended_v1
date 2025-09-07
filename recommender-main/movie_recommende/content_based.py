import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings('ignore')

# =====================================================================================
# == Content-Based Filtering (weighted TF-IDF: Genre=8, Rating=1.5, Title=0.5)
# =====================================================================================

@st.cache_data
def _build_content_model(merged_df: pd.DataFrame):
    """Builds and caches the weighted TF-IDF feature matrix and mappings.
    Features used: Series_Title, Genre, IMDB_Rating.
    Weights: Genre=8.0, Rating=1.5, Title=0.5
    Returns: (feature_matrix, title_to_idx, genre_vectorizer, genre_matrix)
    """
    df = merged_df.copy()
    # Column guards
    genre_col = 'Genre_y' if 'Genre_y' in df.columns else ('Genre_x' if 'Genre_x' in df.columns else 'Genre')
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else ('Rating' if 'Rating' in df.columns else None)

    df['Series_Title'] = df['Series_Title'].astype(str)
    if genre_col not in df.columns:
        df[genre_col] = ''
    df[genre_col] = df[genre_col].fillna('')

    # TF-IDF per field
    genre_vectorizer = TfidfVectorizer(stop_words='english')
    title_vectorizer = TfidfVectorizer(stop_words='english')

    genre_matrix = genre_vectorizer.fit_transform(df[genre_col].astype(str))
    title_matrix = title_vectorizer.fit_transform(df['Series_Title'].astype(str))

    # Rating as numeric feature
    if rating_col is not None and rating_col in df.columns:
        ratings = df[rating_col].fillna(df[rating_col].median()).to_numpy().reshape(-1, 1)
        rating_scaler = MinMaxScaler()
        ratings_scaled = rating_scaler.fit_transform(ratings)  # 0..1
        rating_matrix = csr_matrix(ratings_scaled)
    else:
        rating_matrix = csr_matrix(np.zeros((len(df), 1)))

    # Apply weights
    weighted_genre = genre_matrix.multiply(8.0)
    weighted_title = title_matrix.multiply(0.5)
    weighted_rating = rating_matrix.multiply(1.5)

    # Combine
    feature_matrix = hstack([weighted_genre, weighted_title, weighted_rating]).tocsr()
    title_to_idx = pd.Series(df.index, index=df['Series_Title'].str.lower()).to_dict()

    return feature_matrix, title_to_idx, genre_vectorizer, genre_matrix


def content_based_filtering_enhanced(merged_df: pd.DataFrame, movie_title: str, genre_input: str, top_n: int = 10):
    """Content-based recommendations using weighted TF-IDF and cosine similarity.
    - Features: Genre (8x), Rating (1.5x), Title (0.5x)
    - If movie_title is provided: compute similarity to that movie
    - If only genre_input is provided: rank by genre similarity and rating
    Returns a DataFrame of top N recommendations with a ContentScore column.
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre_x' if 'Genre_x' in merged_df.columns else 'Genre')
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)

    feature_matrix, title_to_idx, genre_vectorizer, genre_matrix = _build_content_model(merged_df)

    # Helper to finalize output
    def _format_output(indices: list, scores: np.ndarray):
        out = merged_df.iloc[indices].copy()
        out = out.drop_duplicates(subset=['Series_Title'])
        out['ContentScore'] = scores[:len(out)]
        # Attach common columns used by UI if present
        keep_cols = ['Series_Title']
        if genre_col in out.columns:
            keep_cols.append(genre_col)
        if rating_col and rating_col in out.columns:
            keep_cols.append(rating_col)
        keep_cols.append('ContentScore')
        return out[keep_cols]

    # Case 1: Movie title provided → similarity to that movie
    if isinstance(movie_title, str) and movie_title.strip():
        key = movie_title.strip().lower()
        if key not in title_to_idx:
            # try loose match
            matches = merged_df[merged_df['Series_Title'].str.lower() == key]
            if matches.empty:
                return pd.DataFrame()
            target_idx = int(matches.index[0])
        else:
            target_idx = int(title_to_idx[key])

        sims = cosine_similarity(feature_matrix[target_idx], feature_matrix).flatten()

        candidates = np.argsort(-sims)
        # Exclude the selected movie itself
        candidates = [i for i in candidates if i != target_idx]

        if isinstance(genre_input, str) and genre_input.strip():
            candidates = [i for i in candidates if genre_input.lower() in str(merged_df.iloc[i][genre_col]).lower()]

        top_indices = candidates[:top_n]
        top_scores = sims[top_indices]
        return _format_output(top_indices, top_scores)

    # Case 2: Only genre provided → similarity to genre query + rating boost
    if isinstance(genre_input, str) and genre_input.strip():
        try:
            genre_vec = genre_vectorizer.transform([genre_input])
            sims = cosine_similarity(genre_vec, genre_matrix).flatten()
        except Exception:
            sims = np.zeros(len(merged_df))

        if rating_col and rating_col in merged_df.columns:
            ratings = merged_df[rating_col].fillna(merged_df[rating_col].median()).to_numpy()
            # Normalize ratings 0..1
            r_scaled = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
            scores = 0.7 * sims + 0.3 * r_scaled
        else:
            scores = sims

        candidates = np.argsort(-scores)[:top_n]
        return _format_output(list(candidates), scores[candidates])

    # No valid input
    return pd.DataFrame()

