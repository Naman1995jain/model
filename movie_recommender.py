import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_movie_data():
    """Load and preprocess movie data"""
    movies = pd.read_csv('Top_rated_movies1.csv')
    movies['overview'] = movies['overview'].fillna('')
    return movies

def create_similarity_matrix(movies):
    """Create TF-IDF and similarity matrix from movie overviews"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def get_movie_recommendations(movie_id, movies, similarity_matrix, n_recommendations=5):
    """Get movie recommendations based on similarity"""
    idx = movies.index[movies['id'] == movie_id].tolist()[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies.iloc[movie_indices][['id', 'title', 'overview', 'vote_average', 'release_date']]
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    return recommendations

def render_movie_recommender():
    """Render the movie recommendation interface"""
    st.title("🎬 Movie Recommendation System")
    
    # Load movie data
    movies = load_movie_data()
    similarity_matrix = create_similarity_matrix(movies)
    
    # Movie selection
    st.subheader("Select a Movie")
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        options=movies['title'].tolist(),
        index=0
    )
    
    if st.button("Get Recommendations", type="primary"):
        movie_id = movies[movies['title'] == selected_movie]['id'].values[0]
        recommendations = get_movie_recommendations(movie_id, movies, similarity_matrix)
        
        st.subheader("Recommended Movies")
        
        for _, movie in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**Rating:** ⭐ {movie['vote_average']:.1f}")
                    st.markdown(f"**Released:** {movie['release_date']}")
                with col2:
                    st.markdown(f"### {movie['title']}")
                    st.markdown(f"_{movie['overview']}_")
                st.markdown("---")

if __name__ == "__main__":
    render_movie_recommender()
