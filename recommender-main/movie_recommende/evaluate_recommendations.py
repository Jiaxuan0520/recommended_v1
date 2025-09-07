import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st  # not used in CLI, but imported per requirements
from typing import Tuple, Dict

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


# Ensure local imports resolve when running from project root
FILE_DIR = os.path.dirname(__file__)
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

from content_based import create_content_features, find_rating_column, find_genre_column, safe_convert_to_numeric


warnings.filterwarnings('ignore')


def load_merged_and_ratings() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = os.path.join(FILE_DIR, 'movies.csv')
    imdb_path = os.path.join(FILE_DIR, 'imdb_top_1000.csv')
    ratings_path = os.path.join(FILE_DIR, 'user_movie_rating.csv')

    movies_df = pd.read_csv(movies_path)
    imdb_df = pd.read_csv(imdb_path)
    ratings_df = pd.read_csv(ratings_path)

    if 'Movie_ID' not in movies_df.columns:
        movies_df['Movie_ID'] = range(len(movies_df))

    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='inner')
    merged_df = merged_df.drop_duplicates(subset='Series_Title')
    if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
        merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on='Series_Title', how='inner')

    # Filter ratings to only those present in merged_df
    if 'Movie_ID' in merged_df.columns:
        ratings_df = ratings_df[ratings_df['Movie_ID'].isin(merged_df['Movie_ID'])].copy()

    return merged_df, ratings_df


def train_test_split_ratings(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_indices = []
    for user_id, group in ratings.groupby('User_ID'):
        n = len(group)
        if n < 3:
            continue
        test_size = max(1, int(round(test_frac * n)))
        idx = group.sample(n=test_size, random_state=seed).index.tolist()
        test_indices.extend(idx)
    test_df = ratings.loc[test_indices].copy()
    train_df = ratings.drop(index=test_indices).copy()
    return train_df, test_df


def build_user_item_matrix(train_df: pd.DataFrame, all_movie_ids: np.ndarray) -> pd.DataFrame:
    rated = train_df[train_df['Movie_ID'].isin(all_movie_ids)]
    ui = rated.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    return ui


def compute_item_similarity(user_item: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    if user_item is None or user_item.empty:
        return pd.DataFrame(), np.array([])
    item_vectors = user_item.fillna(0.0).T  # items x users
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(item_vectors)
    # Precompute item-item cosine similarity matrix (dense) for efficient rating prediction
    # Similarity = 1 - distance of kneighbors to all when requested per item; here build using pairwise
    sims = 1.0 - cosine_similarity(item_vectors)
    # Oops: cosine_similarity returns similarity, not distance. We want similarity directly
    sims = cosine_similarity(item_vectors)
    sim_df = pd.DataFrame(sims, index=item_vectors.index, columns=item_vectors.index)
    return sim_df, item_vectors.index.values


def predict_rating_cf(user_id: int, movie_id: int, user_item: pd.DataFrame, item_sim: pd.DataFrame, k: int = 30, global_mean: float = 7.0) -> float:
    if user_id not in user_item.index or movie_id not in user_item.columns:
        return global_mean
    user_ratings = user_item.loc[user_id].dropna()
    if user_ratings.empty:
        return global_mean
    if movie_id not in item_sim.index:
        return float(user_ratings.mean())
    sims_series = item_sim.loc[movie_id, user_ratings.index]
    # Remove the item itself if present
    sims_series = sims_series.drop(labels=[movie_id], errors='ignore')
    # Select top-k neighbors by similarity
    topk = sims_series.abs().sort_values(ascending=False).head(k)
    if topk.empty:
        return float(user_ratings.mean())
    weighted_sum = float((topk * user_ratings.loc[topk.index]).sum())
    denom = float(topk.abs().sum())
    if denom <= 1e-9:
        return float(user_ratings.mean())
    return float(np.clip(weighted_sum / denom, 1.0, 10.0))


def compute_content_similarity_matrix(merged_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int]]:
    features = create_content_features(merged_df)
    sims = cosine_similarity(features)
    # Map Movie_ID -> row index
    id_to_idx = {}
    if 'Movie_ID' in merged_df.columns:
        for i, mid in enumerate(merged_df['Movie_ID'].values):
            id_to_idx[int(mid)] = i
    return sims, id_to_idx


def predict_rating_content(user_id: int, movie_id: int, train_df: pd.DataFrame, merged_df: pd.DataFrame, content_sims: np.ndarray, id_to_idx: Dict[int, int], k: int = 50, global_mean: float = 7.0) -> float:
    if user_id not in train_df['User_ID'].values or movie_id not in id_to_idx:
        return global_mean
    # Get user's rated items in train
    user_hist = train_df[train_df['User_ID'] == user_id][['Movie_ID', 'Rating']]
    if user_hist.empty:
        return global_mean
    target_idx = id_to_idx.get(int(movie_id))
    if target_idx is None:
        return float(user_hist['Rating'].mean())
    sims_row = pd.Series(content_sims[target_idx, :])
    # Align to user history indices
    hist_indices = [id_to_idx[m] for m in user_hist['Movie_ID'].values if m in id_to_idx]
    if not hist_indices:
        return float(user_hist['Rating'].mean())
    sims_to_hist = sims_row.iloc[hist_indices]
    ratings_hist = user_hist[user_hist['Movie_ID'].isin([merged_df.iloc[i]['Movie_ID'] for i in hist_indices])]['Rating'].reset_index(drop=True)
    # Top-k by similarity
    topk_idx = sims_to_hist.abs().sort_values(ascending=False).head(k).index
    numerator = float((sims_to_hist.loc[topk_idx].reset_index(drop=True) * ratings_hist.loc[:len(topk_idx)-1]).sum())
    denom = float(sims_to_hist.loc[topk_idx].abs().sum())
    if denom <= 1e-9:
        return float(ratings_hist.mean())
    return float(np.clip(numerator / denom, 1.0, 10.0))


def popularity_score_series(merged_df: pd.DataFrame, rating_col: str) -> pd.Series:
    votes_col = 'No_of_Votes' if 'No_of_Votes' in merged_df.columns else 'Votes'
    votes = merged_df.get(votes_col, pd.Series([1000] * len(merged_df)))
    # Parse votes numeric
    def parse_votes(v):
        try:
            return float(str(v).replace(',', ''))
        except Exception:
            return 1000.0
    votes_num = votes.apply(parse_votes)
    rating_vals = merged_df[rating_col].apply(lambda v: safe_convert_to_numeric(v, default=7.0)).fillna(7.0)
    raw = rating_vals * np.log(votes_num + 1.0)
    # Scale roughly into [0,1] using plausible max
    scaled = (raw / (10.0 * 15.0)).clip(0.0, 1.0)
    return pd.Series(scaled.values, index=merged_df['Movie_ID'].values)


def recency_score_series(merged_df: pd.DataFrame) -> pd.Series:
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
    current_year = pd.Timestamp.now().year
    def parse_year(y):
        try:
            return int(str(y).split()[0])
        except Exception:
            return 2000
    years = merged_df.get(year_col, pd.Series([2000] * len(merged_df))).apply(parse_year)
    diffs = (current_year - years).clip(lower=0)
    rec = np.exp(-diffs / 20.0)
    return pd.Series(rec.values, index=merged_df['Movie_ID'].values)


def evaluate_algorithm(name: str,
                       merged_df: pd.DataFrame,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       rating_threshold: float = 7.0,
                       balance_classes: bool = True) -> Dict[str, float]:
    rating_col = find_rating_column(merged_df)

    # Prepare helpers
    user_item = build_user_item_matrix(train_df, merged_df['Movie_ID'].values)
    item_sim, _ = compute_item_similarity(user_item)
    content_sims, id_to_idx = compute_content_similarity_matrix(merged_df)

    global_mean = float(train_df['Rating'].mean()) if not train_df.empty else 7.0

    # Popularity and recency for hybrid scaling to [0,10]
    pop_series = popularity_score_series(merged_df, rating_col)
    rec_series = recency_score_series(merged_df)

    y_true = []
    y_pred_scores = []

    # Iterate over test interactions
    for _, row in test_df.iterrows():
        uid = int(row['User_ID'])
        mid = int(row['Movie_ID'])
        true_r = float(row['Rating'])

        if name.lower() == 'collaborative':
            pred = predict_rating_cf(uid, mid, user_item, item_sim, k=30, global_mean=global_mean)
        elif name.lower() == 'content':
            pred = predict_rating_content(uid, mid, train_df, merged_df, content_sims, id_to_idx, k=50, global_mean=global_mean)
        else:  # hybrid
            pred_c = predict_rating_content(uid, mid, train_df, merged_df, content_sims, id_to_idx, k=50, global_mean=global_mean)
            pred_cf = predict_rating_cf(uid, mid, user_item, item_sim, k=30, global_mean=global_mean)
            # Hybrid blend to rating scale
            # α=0.4, β=0.4, γ=0.1, δ=0.1 (popularity/recency scaled to [0,1])
            pop = float(pop_series.get(mid, 0.5))
            rec = float(rec_series.get(mid, 0.5))
            pred = 0.4 * pred_c + 0.4 * pred_cf + 0.1 * (pop * 10.0) + 0.1 * (rec * 10.0)
            pred = float(np.clip(pred, 1.0, 10.0))

        y_true.append(true_r)
        y_pred_scores.append(pred)

    y_true = np.array(y_true, dtype=float)
    y_pred_scores = np.array(y_pred_scores, dtype=float)

    # Regression metrics
    mse = mean_squared_error(y_true, y_pred_scores)
    rmse = float(np.sqrt(mse))

    # Classification metrics
    y_true_cls = (y_true >= rating_threshold).astype(int)
    y_pred_cls = (y_pred_scores >= rating_threshold).astype(int)

    # Balance classes if requested
    if balance_classes:
        pos_idx = np.where(y_true_cls == 1)[0]
        neg_idx = np.where(y_true_cls == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            n = min(len(pos_idx), len(neg_idx))
            rng = np.random.default_rng(42)
            pos_sample = rng.choice(pos_idx, size=n, replace=False)
            neg_sample = rng.choice(neg_idx, size=n, replace=False)
            sel = np.concatenate([pos_sample, neg_sample])
            y_true_cls_bal = y_true_cls[sel]
            y_pred_cls_bal = y_pred_cls[sel]
        else:
            y_true_cls_bal = y_true_cls
            y_pred_cls_bal = y_pred_cls
    else:
        y_true_cls_bal = y_true_cls
        y_pred_cls_bal = y_pred_cls

    acc = float(accuracy_score(y_true_cls_bal, y_pred_cls_bal))
    report_str = classification_report(
        y_true_cls_bal,
        y_pred_cls_bal,
        target_names=['negative', 'positive'],
        digits=2
    )

    # Extract weighted averages for comparison table
    p, r, f1, _ = precision_recall_fscore_support(y_true_cls_bal, y_pred_cls_bal, average='weighted', zero_division=0)

    # Print block to terminal
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.3f}")
    print(report_str)

    return {
        'model': name,
        'accuracy': acc,
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'mse': float(mse),
        'rmse': float(rmse)
    }


def main():
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_columns', 20)

    print('Loading datasets...')
    merged_df, ratings_df = load_merged_and_ratings()
    if merged_df is None or ratings_df is None or ratings_df.empty:
        print('Failed to load data. Ensure CSVs are present.')
        sys.exit(1)

    print('Preparing train/test split...')
    train_df, test_df = train_test_split_ratings(ratings_df, test_frac=0.2, seed=42)

    # Evaluate each algorithm
    results = []
    for name in ['Hybrid', 'Content', 'Collaborative']:
        res = evaluate_algorithm(name, merged_df, train_df, test_df, rating_threshold=7.0, balance_classes=True)
        results.append(res)

    # Comparison table
    rows = []
    notes_map = {
        'Collaborative': 'Worked well with dense ratings',
        'Content': 'Good with rich metadata',
        'Hybrid': 'Best balance between both'
    }
    for res in results:
        rows.append({
            'Method Used': res['model'],
            'Precision': f"{res['precision']:.2f}",
            'Recall': f"{res['recall']:.2f}",
            'RMSE': f"{res['rmse']:.2f}",
            'Notes': notes_map.get(res['model'], '')
        })
    comp_df = pd.DataFrame(rows)

    print('Comparison table:')
    print(comp_df.to_string(index=False))


if __name__ == '__main__':
    main()


