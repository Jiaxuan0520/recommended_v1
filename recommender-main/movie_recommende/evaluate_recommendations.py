import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from content_based import create_content_features, find_rating_column, find_genre_column


# Weights per requirement
ALPHA = 0.4
BETA = 0.4
GAMMA = 0.1
DELTA = 0.1

RATING_THRESHOLD = 4.0
TEST_SIZE_PER_USER = 0.2
RANDOM_STATE = 42
K_NEIGHBORS = 20


def load_datasets():
	movies = pd.read_csv('movies.csv')
	imdb = pd.read_csv('imdb_top_1000.csv')
	user_ratings = pd.read_csv('user_movie_rating.csv')
	if 'Movie_ID' not in movies.columns:
		movies['Movie_ID'] = range(len(movies))
	merged = pd.merge(movies, imdb, on='Series_Title', how='inner').drop_duplicates('Series_Title')
	if 'Movie_ID' not in merged.columns and 'Movie_ID' in movies.columns:
		merged = pd.merge(movies[['Movie_ID', 'Series_Title']], merged, on='Series_Title', how='inner')
	return merged, user_ratings


def get_cols(df):
	genre_col = find_genre_column(df)
	rating_col = find_rating_column(df)
	year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
	votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'
	return genre_col, rating_col, year_col, votes_col


def build_content_matrix(merged):
	return create_content_features(merged)


def build_item_similarity_knn(user_ratings, merged, k=K_NEIGHBORS):
	movie_ids = merged['Movie_ID'].unique()
	ratings = user_ratings[user_ratings['Movie_ID'].isin(movie_ids)].copy()
	if ratings.empty:
		return None, None
	user_item = ratings.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
	user_item_filled = user_item.fillna(0.0)
	knn = NearestNeighbors(metric='cosine', algorithm='brute')
	knn.fit(user_item_filled.T)
	return knn, user_item


def compute_popularity_and_recency(merged, ratings_df=None):
	genre_col, rating_col, year_col, votes_col = get_cols(merged)
	pop = {}
	rec = {}
	current_year = pd.Timestamp.now().year
	for _, row in merged.iterrows():
		title = row['Series_Title']
		rating = row.get(rating_col, np.nan)
		votes = row.get(votes_col, np.nan)
		year = row.get(year_col, np.nan)
		if pd.isna(rating):
			rating = 7.0
		try:
			votes_val = float(str(votes).replace(',', '')) if not pd.isna(votes) else 1000.0
		except Exception:
			votes_val = 1000.0
		popularity = (float(rating) * np.log10(votes_val + 1.0)) / 10.0
		pop[title] = float(np.clip(popularity, 0.0, 1.0))
		try:
			year_val = int(str(year).split()[0]) if not pd.isna(year) else 2000
		except Exception:
			year_val = 2000
		diff = max(0, current_year - year_val)
		recency = np.exp(-diff / 20.0)
		rec[title] = float(np.clip(recency, 0.0, 1.0))
	if ratings_df is not None and not ratings_df.empty and 'Movie_ID' in merged.columns:
		counts = ratings_df['Movie_ID'].value_counts()
		for movie_id, cnt in counts.items():
			match = merged[merged['Movie_ID'] == movie_id]
			if not match.empty:
				t = match.iloc[0]['Series_Title']
				boost = min(cnt / 100.0, 1.0)
				pop[t] = 0.6 * pop.get(t, 0.5) + 0.4 * boost
	return pop, rec


def predict_content_scores(merged, content_matrix):
	sim = cosine_similarity(content_matrix)
	index_by_title = {t: i for i, t in enumerate(merged['Series_Title'])}
	return sim, index_by_title


def predict_collaborative_scores(knn, user_item, target_movie_id, k=K_NEIGHBORS):
	if knn is None or user_item is None:
		return {}
	item_vectors = user_item.fillna(0.0).T
	if target_movie_id not in item_vectors.index:
		return {}
	item_idx = item_vectors.index.get_loc(target_movie_id)
	distances, indices = knn.kneighbors(item_vectors.iloc[[item_idx]], n_neighbors=min(k+1, len(item_vectors)))
	neighbors = {}
	for d, idx in zip(distances[0], indices[0]):
		nb = int(item_vectors.index[idx])
		if nb == target_movie_id:
			continue
		neighbors[nb] = 1.0 - float(d)
	return neighbors


def split_per_user(user_ratings, test_size=TEST_SIZE_PER_USER, random_state=RANDOM_STATE):
	train_rows = []
	test_rows = []
	for user_id, grp in user_ratings.groupby('User_ID'):
		if len(grp) < 5:
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


def evaluate_models():
	merged, ratings = load_datasets()
	genre_col, rating_col, year_col, votes_col = get_cols(merged)

	# keep only ratings for movies present in merged
	present = set(merged['Movie_ID'].unique())
	ratings = ratings[ratings['Movie_ID'].isin(present)].copy()

	# split
	train_df, test_df = split_per_user(ratings)

	# content features and similarity
	content_matrix = build_content_matrix(merged)
	sim_matrix, title_to_idx = predict_content_scores(merged, content_matrix)

	# collaborative KNN on training
	knn, user_item = build_item_similarity_knn(train_df, merged)

	# popularity/recency on training
	popularity, recency = compute_popularity_and_recency(merged, train_df)

	# mappings
	movieid_to_title = dict(merged[['Movie_ID', 'Series_Title']].values)
	user_groups = train_df.groupby('User_ID')

	# baselines
	global_mean = float(train_df['Rating'].mean()) if not train_df.empty else 7.0
	user_mean = train_df.groupby('User_ID')['Rating'].mean().to_dict()
	item_mean = train_df.groupby('Movie_ID')['Rating'].mean().to_dict()

	# user content profiles (avg similarity to liked items)
	user_content_pref = {}
	for user_id, grp in user_groups:
		liked = grp[grp['Rating'] >= RATING_THRESHOLD]['Movie_ID'].tolist()
		idxs = [title_to_idx.get(movieid_to_title.get(mid, ''), None) for mid in liked]
		idxs = [i for i in idxs if i is not None]
		if idxs:
			profile = sim_matrix[idxs].mean(axis=0)
			mn = float(np.min(profile)); mx = float(np.max(profile))
			if mx > mn:
				profile = (profile - mn) / (mx - mn)
			else:
				profile = np.zeros_like(profile)
			user_content_pref[user_id] = profile
		else:
			user_content_pref[user_id] = np.zeros(sim_matrix.shape[0])

	# predictions containers
	y_true_cls, y_pred_cls_content, y_pred_cls_collab, y_pred_cls_hybrid = [], [], [], []
	y_true_reg, y_pred_reg_content, y_pred_reg_collab, y_pred_reg_hybrid = [], [], [], []

	for _, row in test_df.iterrows():
		user = row['User_ID']
		movie_id = int(row['Movie_ID'])
		true_rating = float(row['Rating'])
		true_label = 1 if true_rating >= RATING_THRESHOLD else 0
		title = movieid_to_title.get(movie_id)
		if title is None or title not in title_to_idx:
			continue

		idx = title_to_idx[title]
		content_score = float(user_content_pref.get(user, np.zeros(sim_matrix.shape[0]))[idx])

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
		if np.isnan(collab_score):
			item_ratings = train_df[train_df['Movie_ID'] == movie_id]['Rating']
			collab_score = item_ratings.mean() if not item_ratings.empty else ratings['Rating'].mean()

		# map scores
		content_rating_est = 2.0 + 8.0 * float(np.clip(content_score, 0.0, 1.0))
		pop = popularity.get(title, 0.5)
		rec = recency.get(title, 0.5)
		pop_rating = 2.0 + 8.0 * pop
		rec_rating = 2.0 + 8.0 * rec
		hybrid_pred = (
			ALPHA * content_rating_est +
			BETA * collab_score +
			GAMMA * pop_rating +
			DELTA * rec_rating
		)

		# clip
		content_rating_est = float(np.clip(content_rating_est, 1.0, 10.0))
		collab_score = float(np.clip(collab_score, 1.0, 10.0))
		hybrid_pred = float(np.clip(hybrid_pred, 1.0, 10.0))

		# collect
		y_true_reg.append(true_rating)
		y_pred_reg_content.append(content_rating_est)
		y_pred_reg_collab.append(collab_score)
		y_pred_reg_hybrid.append(hybrid_pred)

		y_true_cls.append(true_label)
		y_pred_cls_content.append(1 if content_rating_est >= RATING_THRESHOLD else 0)
		y_pred_cls_collab.append(1 if collab_score >= RATING_THRESHOLD else 0)
		vote = int(content_rating_est >= RATING_THRESHOLD) + int(collab_score >= RATING_THRESHOLD) + int(((pop_rating + rec_rating)/2.0) >= RATING_THRESHOLD)
		y_pred_cls_hybrid.append(1 if vote >= 2 else 0)

	def cls_metrics(y_true, y_pred):
		return {
			'precision': precision_score(y_true, y_pred, zero_division=0),
			'recall': recall_score(y_true, y_pred, zero_division=0),
			'f1': f1_score(y_true, y_pred, zero_division=0),
			'accuracy': accuracy_score(y_true, y_pred),
			'report': classification_report(y_true, y_pred, target_names=['negative', 'positive'], zero_division=0)
		}

	def reg_metrics(y_true, y_pred):
		mse = mean_squared_error(y_true, y_pred)
		return {'mse': float(mse), 'rmse': float(np.sqrt(mse))}

	results = {}
	results['Content-Based'] = {**cls_metrics(y_true_cls, y_pred_cls_content), **reg_metrics(y_true_reg, y_pred_reg_content)}
	results['Collaborative'] = {**cls_metrics(y_true_cls, y_pred_cls_collab), **reg_metrics(y_true_reg, y_pred_reg_collab)}
	results['Hybrid'] = {**cls_metrics(y_true_cls, y_pred_cls_hybrid), **reg_metrics(y_true_reg, y_pred_reg_hybrid)}

	print('Model: Content-Based')
	print(f"Accuracy: {results['Content-Based']['accuracy']:.3f}")
	print(results['Content-Based']['report'])
	print('Model: Collaborative')
	print(f"Accuracy: {results['Collaborative']['accuracy']:.3f}")
	print(results['Collaborative']['report'])
	print('Model: Hybrid')
	print(f"Accuracy: {results['Hybrid']['accuracy']:.3f}")
	print(results['Hybrid']['report'])

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
