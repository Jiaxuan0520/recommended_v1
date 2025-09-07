import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def _to_float(value, default: float = 7.0) -> float:
    if pd.isna(value):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        clean = re.sub(r'[^\d.-]', '', str(value))
        return float(clean) if clean != '' else default
    except Exception:
        return default


def find_rating_column(df: pd.DataFrame) -> str:
    return 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'


def find_genre_column(df: pd.DataFrame) -> str:
    return 'Genre_y' if 'Genre_y' in df.columns else 'Genre'


def _build_weighted_text(row: pd.Series, genre_col: str, rating_col: str) -> str:
    # Convert fractional weights (title=0.5, rating=1.5, genre=8) to integers by scaling x2
    TITLE_W = 1   # 0.5 * 2
    GENRE_W = 16  # 8 * 2
    RATING_W = 3  # 1.5 * 2

    title = str(row.get('Series_Title', '')).strip()
    genre = str(row.get(genre_col, '')).strip()
    rating_val = _to_float(row.get(rating_col, np.nan), default=7.0)
    rating_bucket = int(max(1, min(10, round(rating_val))))
    rating_token = f"rating_{rating_bucket}"

    parts = []
    if title:
        parts.extend([title] * TITLE_W)
    if genre:
        parts.extend([genre] * GENRE_W)
    parts.extend([rating_token] * RATING_W)
    return ' '.join(parts)


@st.cache_data
def create_content_features(merged_df: pd.DataFrame):
    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)

    text_series = merged_df.apply(lambda r: _build_weighted_text(r, genre_col, rating_col), axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(text_series)
    return matrix


@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str = None, genre: str = None, top_n: int = 8):
    # If a movie is provided, rank by cosine similarity in the weighted TF-IDF space
    if target_movie and isinstance(target_movie, str) and target_movie.strip():
        titles = merged_df['Series_Title'].astype(str)
        mask = titles.str.lower() == target_movie.strip().lower()
        if not mask.any():
            return None
        target_idx = mask[mask].index[0]

        tfidf_matrix = create_content_features(merged_df)
        target_vec = tfidf_matrix[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        sims = cosine_similarity(target_vec, tfidf_matrix).flatten()

        order = np.argsort(-sims).tolist()
        target_loc = merged_df.index.get_loc(target_idx)
        order = [i for i in order if i != target_loc][:top_n]

        result = merged_df.iloc[order]
        genre_col = find_genre_column(merged_df)
        rating_col = find_rating_column(merged_df)
        return result[['Series_Title', genre_col, rating_col]]

    # If genre query is provided, match using genre-only TF-IDF with the same weights emphasis
    if genre and isinstance(genre, str) and genre.strip():
        genre_col = find_genre_column(merged_df)
        rating_col = find_rating_column(merged_df)

        # Emphasize genre by repeating it; ignore title/rating for query matching
        genre_text = merged_df[genre_col].fillna('').astype(str).apply(lambda g: ' '.join([g] * 16))
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(genre_text)
        q = ' '.join([genre.strip()] * 16)
        qv = vec.transform([q])
        sims = cosine_similarity(qv, mat).flatten()
        order = np.argsort(-sims)[:top_n]
        result = merged_df.iloc[order]
        return result[['Series_Title', genre_col, rating_col]]

    return None