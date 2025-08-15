import streamlit as st
import pickle
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.markdown(
    """
    <style>
    /* Center the labels */
    label {
        display: flex;
        justify-content: center;
        font-weight: bold;
    }
    /* Center the select boxes */
    div[data-baseweb="select"] {
        margin-left: auto;
        margin-right: auto;
        width: 30% !important; /* adjust width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#   SPOTIFY API SETUP
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

@st.cache_data(show_spinner=False)
def get_song_cover(song_name, artist_name):
    """Fetch album cover URL from Spotify API."""
    try:
        query = f"track:{song_name} artist:{artist_name}"
        result = sp.search(q=query, type="track", limit=1)
        if result["tracks"]["items"]:
            return result["tracks"]["items"][0]["album"]["images"][0]["url"]
    except:
        pass
    return None


#   LOAD MODELS & DATA
with open("models/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("models/interaction_matrix.pkl", "rb") as f:
    interaction_matrix = pickle.load(f)

with open("models/kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/data.pkl", "rb") as f:
    data = pickle.load(f)


#   RECOMMENDER FUNCTIONS
def recommend_knn(song_name, k=10):
    # Find playlists containing the song
    song_playlists = data[data['track_name'].str.contains(song_name, case=False, na=False)]['playlist_id'].unique()

    if len(song_playlists) == 0:
        st.warning("Song not found in the dataset.")
        return pd.DataFrame(columns=["track_name", "track_artist"])

    # Get the first playlist ID
    playlist_id = song_playlists[0]

    # Unique playlist IDs
    unique_playlists = data['playlist_id'].unique()

    if playlist_id not in unique_playlists:
        st.warning("Playlist ID not found in unique playlists.")
        return pd.DataFrame(columns=["track_name", "track_artist"])

    # Index of playlist in interaction_matrix
    playlist_index = np.where(unique_playlists == playlist_id)[0][0]

    # Get nearest playlists
    if isinstance(interaction_matrix, pd.DataFrame):
        distances, indices = knn_model.kneighbors(
            interaction_matrix.iloc[playlist_index].values.reshape(1, -1),
            n_neighbors=k+1
        )
    else:
        distances, indices = knn_model.kneighbors(
            interaction_matrix[playlist_index].reshape(1, -1),
            n_neighbors=k+1
        )

    # Remove original playlist
    similar_playlists = indices.flatten()[1:]

    # Collect recommendations
    original_tracks = set(data[data['playlist_id'] == playlist_id]['track_id'])
    recommended_tracks = set()

    for idx in similar_playlists:
        similar_playlist_id = unique_playlists[idx]
        sim_tracks = set(data[data['playlist_id'] == similar_playlist_id]['track_id'])
        recommended_tracks.update(sim_tracks - original_tracks)

    # Return top 10
    if recommended_tracks:
        return data[data['track_id'].isin(recommended_tracks)][['track_name', 'track_artist']].drop_duplicates().head(10)
    else:
        return pd.DataFrame(columns=["track_name", "track_artist"])



def recommend_kmeans(song_name):
    selected_song = data[data['track_name'].str.contains(song_name, case=False, na=False)]
    if selected_song.empty:
        return pd.DataFrame(columns=["track_name", "track_artist"])
    
    cluster = selected_song['cluster_kmeans'].iloc[0]
    recommendations = data[(data['cluster_kmeans'] == cluster) & (data['track_name'] != selected_song['track_name'].iloc[0])]
    return recommendations[['track_name', 'track_artist']].drop_duplicates().head(10)


def display_recommendations_as_cards(df):
    if df.empty:
        st.warning("No recommendations found.")
        return
    
    cols = st.columns(5, gap="medium")  
    for i, (_, row) in enumerate(df.iterrows()):
        cover_url = get_song_cover(row["track_name"], row["track_artist"])
        with cols[i % 5]:
            img_url = cover_url if cover_url else "https://via.placeholder.com/300.png?text=No+Image"
            st.image(img_url, width=300)
            st.markdown(f"**{row['track_name']}**")
            st.caption(row["track_artist"])


#   STREAMLIT UI
st.set_page_config(page_title="🎵 Song Recommendation App",layout='wide')
st.title("🎵 Song Recommendation App")

model_choice = st.selectbox("Select Recommendation Model", ["K Nearest Neighbours", "KMeans Clustering"])
song_choice = st.selectbox(
    "Choose a song",
    sorted(data['track_name'].dropna().unique()),
    index=0,
    placeholder="Search or select a song..."
)

if st.button("Get Recommendations"):
    if model_choice == "K Nearest Neighbours":
        recs = recommend_knn(song_choice)
    else:
        recs = recommend_kmeans(song_choice)
    
    st.write("### Recommended Songs")
    display_recommendations_as_cards(recs)
