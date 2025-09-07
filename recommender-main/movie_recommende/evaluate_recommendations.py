import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
import os
import streamlit as st
warnings.filterwarnings('ignore')

# Import latest content-based helpers and algorithm entry points
from content_based import (
	create_content_features,
	find_rating_column,
	find_genre_column,
	content_based_filtering_enhanced,
)
from collaborative import (
	collaborative_filtering_enhanced,
	load_user_ratings,
)
from hybrid import smart_hybrid_recommendation

# =============================================================
# Configuration
# =============================================================
ALPHA = 0.4  # Content-based weight
BETA = 0.4   # Collaborative weight
GAMMA = 0.1  # Popularity weight
DELTA = 0.1  # Recency weight

# Stricter threshold to prioritize precision
RATING_THRESHOLD = 7.0
TEST_SIZE_PER_USER = 0.2
RANDOM_STATE = 42
K_NEIGHBORS = 20  # for item-based KNN similarity neighborhood

# =============================================================
# Data Loading
# =============================================================


def load_datasets():
	base_dir = os.path.dirname(__file__)
	movies = pd.read_csv(os.path.join(base_dir, 'movies.csv'))
	imdb = pd.read_csv(os.path.join(base_dir, 'imdb_top_1000.csv'))
	user_ratings = pd.read_csv(os.path.join(base_dir, 'user_movie_rating.csv'))
	# ensure Movie_ID exists in merged data
	if 'Movie_ID' not in movies.columns:
		movies['Movie_ID'] = range(len(movies))
	merged = pd.merge(movies, imdb, on='Series_Title', how='inner')
	# Preserve Movie_ID
	if 'Movie_ID' not in merged.columns and 'Movie_ID' in movies.columns:
		merged = pd.merge(movies[['Movie_ID', 'Series_Title']], merged, on='Series_Title', how='inner')
	return merged, user_ratings

# =============================================================
# Helpers
# =============================================================

def get_cols(df):
	genre_col = find_genre_column(df)
	rating_col = find_rating_column(df)
	year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
	votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'
	return genre_col, rating_col, year_col, votes_col


def build_content_matrix(merged):
	# Use the latest content feature builder (title + genre + rating)
	X = create_content_features(merged)
	return X


def build_item_similarity_knn(user_ratings, merged, k=K_NEIGHBORS):
	# Build user-item matrix for available Movie_IDs
	movie_ids = merged['Movie_ID'].unique()
	ratings = user_ratings[user_ratings['Movie_ID'].isin(movie_ids)].copy()
	if ratings.empty:
		return None, None
	user_item = ratings.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
	user_item_filled = user_item.fillna(0.0)
	# Item-based KNN on item vectors (users as features)
	knn = NearestNeighbors(metric='cosine', algorithm='brute')
	knn.fit(user_item_filled.T)
	return knn, user_item


def compute_popularity_and_recency(merged, ratings_df=None):
	_, rating_col, year_col, votes_col = get_cols(merged)
	pop = {}
	rec = {}
	current_year = pd.Timestamp.now().year
	for _, row in merged.iterrows():
		title = row['Series_Title']
		rating = row.get(rating_col, np.nan)
		votes = row.get(votes_col, np.nan)
		year = row.get(year_col, np.nan)
		# Popularity: IMDB_Rating × log(votes); scale roughly to [0,1] to align with hybrid
		if pd.isna(rating):
			rating = 7.0
		if pd.isna(votes) or votes == 0:
			votes = 1000
		votes_val = float(str(votes).replace(',', ''))
		raw_pop = float(rating) * float(np.log(votes_val + 1.0))
		scaled_pop = raw_pop / (10.0 * 15.0)
		pop[title] = float(np.clip(scaled_pop, 0.0, 1.0))
		# Recency: exponential decay by year
		try:
			year_val = int(str(year).split()[0]) if not pd.isna(year) else 2000
		except Exception:
			year_val = 2000
		year_diff = max(0, current_year - year_val)
		recency = np.exp(-year_diff / 20.0)
		rec[title] = float(np.clip(recency, 0.0, 1.0))
	# Interaction-based popularity boost if ratings provided
	if ratings_df is not None and not ratings_df.empty and 'Movie_ID' in merged.columns:
		counts = ratings_df['Movie_ID'].value_counts()
		for movie_id, cnt in counts.items():
			match = merged[merged['Movie_ID'] == movie_id]
			if not match.empty:
				t = match.iloc[0]['Series_Title']
				boost = min(cnt / 100.0, 1.0)
				pop[t] = 0.6 * pop.get(t, 0.5) + 0.4 * boost
	return pop, rec


# =============================================================
# Utility: pick a seed movie per user from training data
# =============================================================

def pick_user_seed(train_df, merged, user_id):
	user_hist = train_df[train_df['User_ID'] == user_id]
	if user_hist.empty:
		return None
	# Choose highest-rated movie in training that exists in merged
	user_hist = user_hist.sort_values(['Rating'], ascending=False)
	ids = user_hist['Movie_ID'].tolist()
	movieid_to_title = dict(merged[['Movie_ID', 'Series_Title']].values)
	for mid in ids:
		if mid in movieid_to_title:
			return movieid_to_title[mid]
	return None


# =============================================================
# Algorithm-connected classification evaluation
# =============================================================

def collect_classification_predictions(merged, train_df, test_df, top_n=8):
	# Make collaborative functions use training data via session_state to avoid leakage
	try:
		st.session_state['user_ratings_df'] = train_df.copy()
	except Exception:
		pass

	movieid_to_title = dict(merged[['Movie_ID', 'Series_Title']].values)
	# Build per-user set of relevant (positive) test titles
	user_pos_titles = {}
	for user_id, grp in test_df.groupby('User_ID'):
		pos = grp[grp['Rating'] >= RATING_THRESHOLD]['Movie_ID'].map(movieid_to_title).dropna().tolist()
		user_pos_titles[user_id] = set(pos)

	algo_names = ['Content-Based', 'Collaborative', 'Hybrid']
	cls_data = {name: {'y_true': [], 'y_pred': []} for name in algo_names}

	all_titles = set(merged['Series_Title'].tolist())

	for user_id in sorted(test_df['User_ID'].unique()):
		seed_title = pick_user_seed(train_df, merged, user_id)
		if not seed_title:
			continue
		relevant = user_pos_titles.get(user_id, set())
		if len(relevant) == 0:
			continue

		# Content-Based recommendations
		cb_df = content_based_filtering_enhanced(merged, seed_title, genre=None, top_n=top_n)
		cb_recs = cb_df['Series_Title'].tolist() if cb_df is not None and not cb_df.empty else []

		# Collaborative recommendations
		cf_df = collaborative_filtering_enhanced(merged, seed_title, top_n=top_n)
		cf_recs = cf_df['Series_Title'].tolist() if cf_df is not None and not cf_df.empty else []

		# Hybrid recommendations
		hb_df = smart_hybrid_recommendation(merged, seed_title, genre=None, top_n=top_n)
		hb_recs = hb_df['Series_Title'].tolist() if hb_df is not None and not hb_df.empty else []

		for name, recs in [('Content-Based', cb_recs), ('Collaborative', cf_recs), ('Hybrid', hb_recs)]:
			if not recs:
				continue
			# Positive predictions: recommended items (label by membership in relevant set)
			y_true_pos = [1 if t in relevant else 0 for t in recs]
			y_pred_pos = [1] * len(recs)
			# Negative predictions: sample same number of non-recommended, non-relevant items
			neg_pool = list(all_titles - set(recs) - set(relevant))
			if len(neg_pool) >= len(recs):
				neg_sample = np.random.RandomState(42).choice(neg_pool, size=len(recs), replace=False)
			else:
				neg_sample = neg_pool
			y_true_neg = [0] * len(neg_sample)
			y_pred_neg = [0] * len(neg_sample)
			cls_data[name]['y_true'].extend(y_true_pos + y_true_neg)
			cls_data[name]['y_pred'].extend(y_pred_pos + y_pred_neg)

	return cls_data

# =============================================================
# Predictors
# =============================================================

def predict_content_scores(merged, content_matrix):
	# Return raw cosine similarity scores between items (already in [0,1] for TF-IDF)
	sim = cosine_similarity(content_matrix)
	index_by_title = {t: i for i, t in enumerate(merged['Series_Title'])}
	return sim, index_by_title


def predict_collaborative_scores(knn, user_item, target_movie_id, k=K_NEIGHBORS):
	# For an item, get its k nearest neighbors by cosine distance
	# Return dict: neighbor_movie_id -> similarity
	if knn is None or user_item is None:
		return {}
	item_vectors = user_item.fillna(0.0).T
	if target_movie_id not in item_vectors.index:
		return {}
	item_idx = item_vectors.index.get_loc(target_movie_id)
	distances, indices = knn.kneighbors(item_vectors.iloc[[item_idx]], n_neighbors=min(k+1, len(item_vectors)))
	neighbors = {}
	for d, idx in zip(distances[0], indices[0]):
		neighbor_movie = item_vectors.index[idx]
		if neighbor_movie == target_movie_id:
			continue
		neighbors[int(neighbor_movie)] = 1.0 - float(d)  # cosine similarity
	return neighbors


# =============================================================
# Train/Test Split per user
# =============================================================

def split_per_user(user_ratings, test_size=TEST_SIZE_PER_USER, random_state=RANDOM_STATE):
	train_rows = []
	test_rows = []
	for user_id, grp in user_ratings.groupby('User_ID'):
		if len(grp) < 5:
			# small history: simple split
			grp_shuffled = grp.sample(frac=1, random_state=random_state)
			split_idx = int(len(grp_shuffled) * (1 - test_size))
			train_rows.append(grp_shuffled.iloc[:split_idx])
			test_rows.append(grp_shuffled.iloc[split_idx:])
		else:
			tr, te = train_test_split(grp, test_size=test_size, random_state=random_state)
			train_rows.append(tr)
			test_rows.append(te)
	train_df = pd.concat(train_rows).reset_index(drop=True)
	test_df = pd.concat(test_rows).reset_index(drop=True)
	return train_df, test_df

# =============================================================
# Evaluation Pipeline
# =============================================================

def evaluate_models():
	merged, ratings = load_datasets()
	genre_col, rating_col, year_col, votes_col = get_cols(merged)

	# Filter ratings to those movies present in merged
	present_ids = set(merged['Movie_ID'].unique())
	ratings = ratings[ratings['Movie_ID'].isin(present_ids)].copy()

	# Split BEFORE building models to avoid leakage
	train_df, test_df = split_per_user(ratings)

	# Content features/similarity (independent of split)
	content_matrix = build_content_matrix(merged)
	sim_matrix, title_to_idx = predict_content_scores(merged, content_matrix)

	# Collaborative KNN built on training data only
	knn, user_item = build_item_similarity_knn(train_df, merged)

	# Popularity/recency based on training interactions
	popularity, recency = compute_popularity_and_recency(merged, train_df)

	# Build quick lookups
	movieid_to_title = dict(merged[['Movie_ID', 'Series_Title']].values)
	title_to_movieid = {v: k for k, v in movieid_to_title.items()}
	user_train = train_df.groupby('User_ID')

	# Baselines for collaborative filtering (bias terms)
	global_mean = float(train_df['Rating'].mean()) if not train_df.empty else 7.0
	user_mean = train_df.groupby('User_ID')['Rating'].mean().to_dict()
	item_mean = train_df.groupby('Movie_ID')['Rating'].mean().to_dict()

	# Optional: make collaborative functions (if called) read training ratings
	try:
		st.session_state['user_ratings_df'] = train_df.copy()
	except Exception:
		pass

	# Predictions and ground truth (rating-based classification per test row)
	y_true_cls = []
	y_pred_cls_content = []
	y_pred_cls_collab = []
	y_pred_cls_hybrid = []

	y_true_reg = []
	y_pred_reg_content = []
	y_pred_reg_collab = []
	y_pred_reg_hybrid = []

	# Iterate over test set rows for rating prediction using training data only
	for _, row in test_df.iterrows():
		user = row['User_ID']
		movie_id = int(row['Movie_ID'])
		true_rating = float(row['Rating'])
		true_label = 1 if true_rating >= RATING_THRESHOLD else 0
		title = movieid_to_title.get(movie_id)
		if title is None or title not in title_to_idx:
			# skip if not in merged dataset
			continue

		# Content-based rating via similarity to user's training ratings
		target_idx = title_to_idx[title]
		user_train_rows = train_df[train_df['User_ID'] == user]
		content_pred = np.nan
		if not user_train_rows.empty:
			# Build neighbors from items the user rated in training
			sim_vals = []
			ratings_vals = []
			for _, tr in user_train_rows.iterrows():
				mid_j = int(tr['Movie_ID'])
				if mid_j in movieid_to_title:
					title_j = movieid_to_title[mid_j]
					if title_j in title_to_idx:
						idx_j = title_to_idx[title_j]
						s = sim_matrix[idx_j, target_idx]
						sim_vals.append(max(0.0, float(s)))
						ratings_vals.append(float(tr['Rating']))
			if len(sim_vals) > 0 and np.sum(sim_vals) > 0:
				content_pred = float(np.dot(sim_vals, ratings_vals) / np.sum(sim_vals))
		if np.isnan(content_pred):
			# Fallbacks
			content_pred = item_mean.get(movie_id, global_mean)
		# Collaborative score: baseline-corrected item-based KNN (r_hat = b_u + b_i + sum sim*(r_uj - b_u - b_j)/sum|sim|)
		neighbor_sims = predict_collaborative_scores(knn, user_item, movie_id, k=K_NEIGHBORS)
		b_u = user_mean.get(user, global_mean)
		b_i = item_mean.get(movie_id, global_mean)
		numerator = 0.0
		norm = 0.0
		if neighbor_sims:
			user_row = user_item.loc[user].dropna() if (user in user_item.index) else pd.Series(dtype=float)
			for nb_movie, sim in neighbor_sims.items():
				if nb_movie in user_row.index:
					r_uj = float(user_row.loc[nb_movie])
					b_j = item_mean.get(int(nb_movie), global_mean)
					numerator += sim * (r_uj - b_u - b_j)
					norm += abs(sim)
			if norm > 0:
				collab_score = b_u + b_i + (numerator / norm)
			else:
				collab_score = np.nan
		else:
			collab_score = np.nan
		# Fallback to item mean if no info
		if np.isnan(collab_score):
			item_ratings = train_df[train_df['Movie_ID'] == movie_id]['Rating']
			collab_score = item_ratings.mean() if not item_ratings.empty else ratings['Rating'].mean()

		content_rating_est = float(np.clip(content_pred, 1.0, 10.0))

		# Popularity & Recency (0..1) mapped to 2..10
		pop = popularity.get(title, 0.5)
		rec = recency.get(title, 0.5)
		pop_rating = 2.0 + 8.0 * pop
		rec_rating = 2.0 + 8.0 * rec

		# Hybrid final score (rating prediction)
		hybrid_pred = (
			ALPHA * content_rating_est +
			BETA * collab_score +
			GAMMA * pop_rating +
			DELTA * rec_rating
		)

		# Clip to rating bounds
		content_rating_est = float(np.clip(content_rating_est, 1.0, 10.0))
		collab_score = float(np.clip(collab_score, 1.0, 10.0))
		hybrid_pred = float(np.clip(hybrid_pred, 1.0, 10.0))

		# Collect regression targets
		y_true_reg.append(true_rating)
		y_pred_reg_content.append(content_rating_est)
		y_pred_reg_collab.append(collab_score)
		y_pred_reg_hybrid.append(hybrid_pred)

		# Collect classification decisions per algorithm using predicted ratings
		y_true_cls.append(true_label)
		y_pred_cls_content.append(1 if content_rating_est >= RATING_THRESHOLD else 0)
		y_pred_cls_collab.append(1 if collab_score >= RATING_THRESHOLD else 0)
		y_pred_cls_hybrid.append(1 if hybrid_pred >= RATING_THRESHOLD else 0)

	# Compute metrics on a balanced subset to avoid class imbalance bias
	def compute_classification_metrics(y_true, y_pred):
		# balance
		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)
		pos_idx = np.where(y_true == 1)[0]
		neg_idx = np.where(y_true == 0)[0]
		if len(pos_idx) == 0 or len(neg_idx) == 0:
			return {
				'precision': 0.0,
				'recall': 0.0,
				'f1': 0.0,
				'accuracy': 0.0,
				'report': 'Insufficient positive/negative samples'
			}
		m = min(len(pos_idx), len(neg_idx))
		rng = np.random.RandomState(42)
		pos_sample = rng.choice(pos_idx, size=m, replace=False)
		neg_sample = rng.choice(neg_idx, size=m, replace=False)
		idx = np.concatenate([pos_sample, neg_sample])
		yt = y_true[idx]
		yp = y_pred[idx]
		return {
			'precision': precision_score(yt, yp, zero_division=0),
			'recall': recall_score(yt, yp, zero_division=0),
			'f1': f1_score(yt, yp, zero_division=0),
			'accuracy': accuracy_score(yt, yp),
			'report': classification_report(yt, yp, target_names=['negative', 'positive'], zero_division=0)
		}

	def compute_regression_metrics(y_true, y_pred):
		mse = mean_squared_error(y_true, y_pred)
		return {'mse': mse, 'rmse': float(np.sqrt(mse))}

	results = {}
	# Use each method's own y_true from cls_data for fair classification metrics
	results['Content-Based'] = {
		**compute_classification_metrics(cls_data['Content-Based']['y_true'], cls_data['Content-Based']['y_pred']),
		**compute_regression_metrics(y_true_reg, y_pred_reg_content)
	}
	results['Collaborative'] = {
		**compute_classification_metrics(cls_data['Collaborative']['y_true'], cls_data['Collaborative']['y_pred']),
		**compute_regression_metrics(y_true_reg, y_pred_reg_collab)
	}
	results['Hybrid'] = {
		**compute_classification_metrics(cls_data['Hybrid']['y_true'], cls_data['Hybrid']['y_pred']),
		**compute_regression_metrics(y_true_reg, y_pred_reg_hybrid)
	}

	# Display
	print('Model: Content-Based')
	print(f"Accuracy: {results['Content-Based']['accuracy']:.3f}")
	print(results['Content-Based']['report'])
	print('Model: Collaborative')
	print(f"Accuracy: {results['Collaborative']['accuracy']:.3f}")
	print(results['Collaborative']['report'])
	print('Model: Hybrid')
	print(f"Accuracy: {results['Hybrid']['accuracy']:.3f}")
	print(results['Hybrid']['report'])

	# Summary table
	summary_rows = []
	for name in ['Collaborative', 'Content-Based', 'Hybrid']:
		row = {
			'Method Used': name,
			'Precision': round(results[name]['precision'], 2),
			'Recall': round(results[name]['recall'], 2),
			'RMSE': round(results[name]['rmse'], 2),
			'Notes': (
				'Worked well with dense ratings' if name == 'Collaborative' else
				'Good with rich metadata' if name == 'Content-Based' else
				'Best balance between both'
			)
		}
		summary_rows.append(row)
	summary_df = pd.DataFrame(summary_rows, columns=['Method Used', 'Precision', 'Recall', 'RMSE', 'Notes'])
	print('\nComparison Table:')
	print(summary_df.to_string(index=False))


if __name__ == '__main__':
	evaluate_models()
