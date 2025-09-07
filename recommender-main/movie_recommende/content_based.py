import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import re
from difflib import get_close_matches
import streamlit as st


def safe_convert_to_numeric(value, default=None):
    """Safely convert a value to numeric, handling strings and NaN"""
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove any non-numeric characters except decimal point
        clean_value = re.sub(r'[^\d.-]', '', str(value))
        try:
            return float(clean_value) if clean_value else default
        except (ValueError, TypeError):
            return default
    
    return default


def find_rating_column(df: pd.DataFrame) -> str:
    return 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'


def find_genre_column(df: pd.DataFrame) -> str:
    return 'Genre_y' if 'Genre_y' in df.columns else 'Genre'


def _get_title_series(df: pd.DataFrame) -> pd.Series:
    return df['Series_Title'].fillna('').astype(str)


def find_similar_titles(input_title, titles_list, cutoff=0.6):
    """Enhanced fuzzy matching for movie titles"""
    if not input_title or not titles_list:
        return []
    
    input_lower = input_title.lower().strip()
    
    # Direct match
    exact_matches = [title for title in titles_list if title.lower() == input_lower]
    if exact_matches:
        return exact_matches
    
    # Partial match
    partial_matches = []
    for title in titles_list:
        title_lower = title.lower()
        if input_lower in title_lower:
            partial_matches.append((title, len(input_lower) / len(title_lower)))
        elif title_lower in input_lower:
            partial_matches.append((title, len(title_lower) / len(input_lower)))
            
    if partial_matches:
        # Sort by match ratio
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in partial_matches]

    # Close matches
    return get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)


@st.cache_data
def create_content_features(merged_df: pd.DataFrame):
    """Create weighted content feature matrix using TF-IDF and numeric rating.

    Features and weights (as requested):
    - Series_Title (TF-IDF) weight = 0.5
    - Genre (TF-IDF) weight = 8.0
    - IMDB_Rating (numeric scaled to [0,1]) weight = 1.5
    """
    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)

    # Text corpora
    title_text = _get_title_series(merged_df)
    genre_text = merged_df[genre_col].fillna('').astype(str)

    # Vectorize title and genre separately
    title_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    genre_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)

    title_tfidf = title_vec.fit_transform(title_text)
    genre_tfidf = genre_vec.fit_transform(genre_text)

    # Weights
    TITLE_W = 0.5
    GENRE_W = 8.0
    RATING_W = 1.5

    # Rating numeric column → scaled to [0,1] (divide by 10), fill with median or 7.0 fallback
    ratings = merged_df[rating_col].apply(lambda v: safe_convert_to_numeric(v, default=np.nan))
    if ratings.isna().all():
        ratings = pd.Series([7.0] * len(merged_df), index=merged_df.index)
    ratings_filled = ratings.fillna(ratings.median() if not pd.isna(ratings.median()) else 7.0)
    rating_norm = (ratings_filled.clip(lower=0.0, upper=10.0) / 10.0) * RATING_W
    rating_csr = csr_matrix(rating_norm.values.reshape(-1, 1))

    # Weighted concatenation
    weighted_title = title_tfidf * TITLE_W
    weighted_genre = genre_tfidf * GENRE_W
    content_features = hstack([weighted_title, weighted_genre, rating_csr], format='csr')
    return content_features


@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=8):
    """Content-Based filtering using weighted TF-IDF on title, genre, plus numeric rating (cosine similarity)."""
    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)

    # Target movie mode
    if target_movie and str(target_movie).strip():
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        target_title = similar_titles[0]
        if target_title not in merged_df['Series_Title'].values:
            return None

        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        features = create_content_features(merged_df)
        target_loc = merged_df.index.get_loc(target_idx)
        sims = cosine_similarity(features[target_loc], features).flatten()

        order = list(np.argsort(-sims))
        # drop self index
        order = [i for i in order if i != target_loc]
        # optional genre filtering
        if genre and str(genre).strip():
            mask = merged_df[genre_col].fillna('').str.contains(str(genre), case=False, regex=False)
            order = [i for i in order if bool(mask.iloc[i])]
        top_idx = order[:top_n]
        result_df = merged_df.iloc[top_idx]
        return result_df[['Series_Title', genre_col, rating_col]]

    # Genre query mode (no target movie)
    if genre and str(genre).strip():
        genre_corpus = merged_df[genre_col].fillna('').astype(str).tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query = tfidf.transform([str(genre)])
        sims = cosine_similarity(query, tfidf_matrix).flatten()
        order = np.argsort(-sims)[:top_n]
        result_df = merged_df.iloc[order]
        return result_df[['Series_Title', genre_col, rating_col]]

    return None