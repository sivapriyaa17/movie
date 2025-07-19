
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    ratings = pd.read_csv(
        r"C:\Users\Haripriya\Downloads\ml-100k\ml-100k\u.data",
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

    movies = pd.read_csv(
        r"C:\Users\Haripriya\Downloads\ml-100k\ml-100k\u.item",
        sep='|',
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    )
    return ratings, movies


def get_cf_recommendations(user_id, ratings_matrix, movie_titles, top_n=5):
    if user_id not in ratings_matrix.index:
        return ["User ID not found in dataset."]
    
    user_vector = ratings_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, ratings_matrix)[0]
    similar_users = list(enumerate(similarity_scores))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]

    weighted_scores = np.zeros(ratings_matrix.shape[1])
    sim_sum = 0
    for user_idx, score in similar_users[:10]:
        weighted_scores += ratings_matrix.iloc[user_idx] * score
        sim_sum += score

    if sim_sum > 0:
        weighted_scores /= sim_sum

    user_seen = ratings_matrix.loc[user_id] > 0
    unseen_scores = [(i, score) for i, score in enumerate(weighted_scores) if not user_seen[i]]
    unseen_scores = sorted(unseen_scores, key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = [movie_titles[i] for i, _ in unseen_scores]
    return recommendations


def get_content_recommendations(movie_title, movies, similarity_content, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found."]
    
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores[:top_n]]
    
    return movies.iloc[movie_indices]['title'].tolist()


st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")

ratings, movies = load_data()


data = pd.merge(ratings, movies, on="movie_id")
ratings_matrix = data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
movie_titles = ratings_matrix.columns.tolist()
genre_cols = movies.columns[5:]
genre_matrix = movies[genre_cols].values
similarity_content = cosine_similarity(genre_matrix)


option = st.selectbox("Choose Recommendation Type:", ["Collaborative Filtering", "Content-Based"])

if option == "Collaborative Filtering":
    user_id = st.slider("Select User ID", 1, 943)
    
    if st.button("Get Recommendations"):
        recs = get_cf_recommendations(user_id, ratings_matrix, movie_titles)
        st.write("### Top 5 Recommendations for You:")
        for title in recs:
            st.markdown(f"- {title}")

elif option == "Content-Based":
    movie_list = sorted(movies['title'].dropna().unique())
    movie_title = st.selectbox("Select a Movie", movie_list)
    
    if st.button("Find Similar Movies"):
        if movie_title:
            recs = get_content_recommendations(movie_title, movies, similarity_content)
            st.write(f"### Top 5 Movies Similar to *{movie_title}*:")
            for title in recs:
                st.markdown(f"- {title}")
        else:
            st.warning("⚠️ Please select a movie.")