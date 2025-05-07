import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("music.csv")

# Drop missing or duplicate values
df = df.dropna(subset=["track_id", "track_name", "track_genre"])
df = df.drop_duplicates(subset=["track_id"])

# TF-IDF on genre column
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["track_genre"])

# Cosine similarity between tracks
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Track index map
indices = pd.Series(df.index, index=df["track_name"]).drop_duplicates()

def recommend_tracks(track_name, num=5):
    if track_name not in indices:
        return []
    idx = indices[track_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    track_indices = [i[0] for i in sim_scores]
    return df.iloc[track_indices][["track_name", "track_genre", "popularity"]]