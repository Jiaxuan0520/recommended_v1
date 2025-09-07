import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================================================
# == Hybrid Recommender for Streamlit (Linear Weighted Blend)
# =====================================================================================

@st.cache_data
def _compute_popularity_and_recency(merged_df: pd.DataFrame):
    """Compute normalized popularity and recency scores for all movies.
    Popularity = IMDB_Rating * log(No_of_Votes)
    Recency = exp(-(current_year - Released_Year)/k)
    Returns two pd.Series indexed like merged_df.
    """
    df = merged_df.copy()
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else ('Rating' if 'Rating' in df.columns else None)
    votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else None
    year_col = 'Released_Year' if 'Released_Year' in df.columns else ('Year' if 'Year' in df.columns else None)

    # Popularity
    if rating_col and votes_col and votes_col in df.columns:
        votes = pd.to_numeric(df[votes_col], errors='coerce').fillna(0.0)
        ratings = pd.to_numeric(df[rating_col], errors='coerce').fillna(df[rating_col].median() if rating_col in df.columns else 0.0)
        popularity = ratings * np.log1p(votes)
    else:
        popularity = pd.Series(np.zeros(len(df)), index=df.index)

    # Recency
    if year_col and year_col in df.columns:
        year_values = pd.to_numeric(df[year_col], errors='coerce').fillna(df[year_col].median() if year_col in df.columns else 2000)
        current_year = datetime.now().year
        k = 15.0
        recency = np.exp(-(current_year - year_values) / k)
    else:
        recency = pd.Series(np.zeros(len(df)), index=df.index)

    # Normalize 0..1
    def _normalize(series: pd.Series):
        arr = series.to_numpy(dtype=float)
        min_v, max_v = np.nanmin(arr), np.nanmax(arr)
        if max_v - min_v < 1e-9:
            return pd.Series(np.zeros_like(arr), index=series.index)
        return pd.Series((arr - min_v) / (max_v - min_v + 1e-9), index=series.index)

    pop_norm = _normalize(popularity)
    rec_norm = _normalize(recency)
    return pop_norm, rec_norm


def smart_hybrid_recommendation(merged_df: pd.DataFrame, movie_title: str, genre_input: str, top_n: int = 10):
    """Hybrid recommendations combining Content (0.4) + Collaborative (0.4) + Popularity (0.1) + Recency (0.1).
    Returns a DataFrame suitable for display in the Streamlit app.
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre_x' if 'Genre_x' in merged_df.columns else 'Genre')
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)

    # Content-based candidates
    content_df = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n=max(top_n * 4, 20))
    if content_df is None or content_df.empty:
        # Fallback to collaborative only
        cf_df = collaborative_filtering_enhanced(merged_df, movie_title, top_n=top_n)
        return cf_df if cf_df is not None else pd.DataFrame()
    if 'ContentScore' not in content_df.columns:
        # If content function didn't attach a score, create a simple rank score
        content_df = content_df.copy()
        content_df['ContentScore'] = np.linspace(1.0, 0.0, num=len(content_df), endpoint=False)

    # Collaborative scores for same anchor movie
    cf_df = collaborative_filtering_enhanced(merged_df, movie_title, top_n=max(top_n * 4, 20))
    if cf_df is None:
        cf_df = pd.DataFrame(columns=['Series_Title', 'Similarity'])
    if 'Similarity' not in cf_df.columns:
        cf_df['Similarity'] = 0.0

    # Merge by title
    merged_scores = content_df.merge(
        cf_df[['Series_Title', 'Similarity']], on='Series_Title', how='left'
    )
    merged_scores['Similarity'] = merged_scores['Similarity'].fillna(0.0)

    # Popularity and Recency
    pop_norm, rec_norm = _compute_popularity_and_recency(merged_df)
    aux = merged_df[['Series_Title']].copy()
    aux['pop_norm'] = pop_norm.values
    aux['rec_norm'] = rec_norm.values
    merged_scores = merged_scores.merge(aux, on='Series_Title', how='left')
    merged_scores[['pop_norm', 'rec_norm']] = merged_scores[['pop_norm', 'rec_norm']].fillna(0.0)

    # Normalize Content and CF to 0..1
    normalizer = MinMaxScaler()
    merged_scores[['ContentScore', 'Similarity']] = normalizer.fit_transform(
        merged_scores[['ContentScore', 'Similarity']]
    )

    # Weighted blend
    merged_scores['FinalScore'] = (
        0.4 * merged_scores['ContentScore'] +
        0.4 * merged_scores['Similarity'] +
        0.1 * merged_scores['pop_norm'] +
        0.1 * merged_scores['rec_norm']
    )

    # Remove the anchor movie if it's present
    if isinstance(movie_title, str) and movie_title.strip():
        merged_scores = merged_scores[merged_scores['Series_Title'].str.lower() != movie_title.strip().lower()]

    # If genre filter provided, prefer matching ones slightly
    if isinstance(genre_input, str) and genre_input.strip() and genre_col in merged_df.columns:
        genre_boost_titles = set(
            merged_df[merged_df[genre_col].fillna('').str.lower().str.contains(genre_input.strip().lower())]['Series_Title']
        )
        merged_scores['FinalScore'] += merged_scores['Series_Title'].isin(genre_boost_titles) * 0.02

    # Attach display columns
    display_cols = ['Series_Title']
    if genre_col in merged_df.columns:
        display_cols.append(genre_col)
    if rating_col and rating_col in merged_df.columns:
        display_cols.append(rating_col)

    display_df = merged_df[display_cols].drop_duplicates('Series_Title')
    merged_scores = merged_scores.merge(display_df, on='Series_Title', how='left')

    merged_scores = merged_scores.sort_values('FinalScore', ascending=False)
    return merged_scores.head(top_n)


