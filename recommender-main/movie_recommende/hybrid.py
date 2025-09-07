import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from content_based import create_content_features, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings


class LinearHybridRecommender:
    def __init__(self, merged_df: pd.DataFrame):
        self.merged_df = merged_df
        self.rating_col = find_rating_column(merged_df)
        self.genre_col = find_genre_column(merged_df)
        self.user_ratings_df = load_user_ratings()
        # Weights per requirement
        self.alpha = 0.4
        self.beta = 0.4
        self.gamma = 0.1
        self.delta = 0.1

    def _content_scores(self, target_movie: str, genre: str, top_n: int) -> dict:
        scores = {}
        if target_movie and isinstance(target_movie, str) and target_movie.strip():
            tfidf = create_content_features(self.merged_df)
            titles = self.merged_df['Series_Title'].astype(str)
            mask = titles.str.lower() == target_movie.strip().lower()
            if not mask.any():
                return scores
            target_idx = mask[mask].index[0]
            target_loc = self.merged_df.index.get_loc(target_idx)
            sims = cosine_similarity(tfidf[target_loc], tfidf).flatten()
            order = [i for i in np.argsort(-sims) if i != target_loc][: top_n * 3]
            for i in order:
                scores[self.merged_df.iloc[i]['Series_Title']] = float(sims[i])
            return scores
        if genre and isinstance(genre, str) and genre.strip():
            genre_col = self.genre_col
            # Genre-only match using cosine similarity
            genre_text = self.merged_df[genre_col].fillna('').astype(str).apply(lambda g: ' '.join([g] * 16))
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            mat = vec.fit_transform(genre_text)
            q = ' '.join([genre.strip()] * 16)
            qv = vec.transform([q])
            sims = cosine_similarity(qv, mat).flatten()
            order = np.argsort(-sims)[: top_n * 3]
            for i in order:
                scores[self.merged_df.iloc[i]['Series_Title']] = float(sims[i])
            return scores
        return scores

    def _collab_scores(self, target_movie: str, top_n: int) -> dict:
        scores = {}
        if target_movie and self.user_ratings_df is not None:
            res = collaborative_knn(self.merged_df, target_movie, top_n=top_n * 3)
            if res is not None and not res.empty:
                if 'Similarity' in res.columns:
                    for _, r in res.iterrows():
                        scores[r['Series_Title']] = float(r['Similarity'])
                else:
                    for _, r in res.iterrows():
                        scores[r['Series_Title']] = 1.0
        return scores

    def _popularity_scores(self) -> dict:
        scores = {}
        votes_col = 'No_of_Votes' if 'No_of_Votes' in self.merged_df.columns else 'Votes'
        for _, row in self.merged_df.iterrows():
            title = row['Series_Title']
            rating = row.get(self.rating_col, 7.0)
            votes = row.get(votes_col, 1000)
            try:
                v = float(str(votes).replace(',', ''))
            except Exception:
                v = 1000.0
            r = 7.0 if pd.isna(rating) else float(rating)
            pop = (r * np.log10(v + 1.0)) / 10.0
            scores[title] = float(np.clip(pop, 0.0, 1.0))
        return scores

    def _recency_scores(self) -> dict:
        scores = {}
        year_col = 'Released_Year' if 'Released_Year' in self.merged_df.columns else 'Year'
        current_year = pd.Timestamp.now().year
        for _, row in self.merged_df.iterrows():
            title = row['Series_Title']
            year = row.get(year_col, 2000)
            try:
                y = int(str(year).split()[0]) if not pd.isna(year) else 2000
            except Exception:
                y = 2000
            diff = max(0, current_year - y)
            rec = np.exp(-diff / 20.0)
            scores[title] = float(np.clip(rec, 0.0, 1.0))
        return scores

    def recommend(self, target_movie: str = None, genre: str = None, top_n: int = 8):
        content_scores = self._content_scores(target_movie, genre, top_n)
        collab_scores = self._collab_scores(target_movie, top_n)
        popularity_scores = self._popularity_scores()
        recency_scores = self._recency_scores()

        candidates = set(content_scores) | set(collab_scores)
        if len(candidates) < top_n * 2:
            for t, _ in sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]:
                candidates.add(t)

        final_scores = {}
        for t in candidates:
            c = content_scores.get(t, 0.0)
            cf = collab_scores.get(t, 0.0)
            pop = popularity_scores.get(t, 0.5)
            rec = recency_scores.get(t, 0.5)
            final_scores[t] = float(self.alpha * c + self.beta * cf + self.gamma * pop + self.delta * rec)

        top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        titles = [t for t, _ in top]
        result = self.merged_df[self.merged_df['Series_Title'].isin(titles)]
        if result.empty:
            return None
        order = {t: i for i, t in enumerate(titles)}
        result = result.copy()
        result['rank_order'] = result['Series_Title'].map(order)
        result = result.sort_values('rank_order').drop(columns=['rank_order'])
        return result[['Series_Title', self.genre_col, self.rating_col]]


@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    return LinearHybridRecommender(merged_df).recommend(target_movie, genre, top_n)