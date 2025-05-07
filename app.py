import streamlit as st
from model import recommend_tracks, df

st.title("🎵 Music Recommender")
st.markdown("Select a track and get similar music recommendations based on genre.")

track_list = df["track_name"].dropna().unique()
selected_track = st.selectbox("Choose a track", sorted(track_list))

if st.button("Recommend"):
    results = recommend_tracks(selected_track)
    if not results.empty:
        st.write("### Recommended Tracks:")
        st.dataframe(results)
    else:
        st.warning("No recommendations found.")
