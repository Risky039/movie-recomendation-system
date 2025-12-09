# movie_recommender_app.py
import streamlit as st
import requests
import pandas as pd
import json
import os
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

# ====== IMPORTS ======
import os
import json
import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data for sentiment analysis (run this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

#CONFIG
TMDB_API_KEY = "6c60cf0bb750a16abf3574dfeafcbfc9"
TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
WATCHLIST_FILE = "watchlists.json"

# ====== UTILITIES ======
def tmdb_get(path, params=None):
    if params is None:
        params = {}
    params.update({"api_key": TMDB_API_KEY})
    resp = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def search_movie(query, page=1):
    return tmdb_get("/search/movie", {"query": query, "page": page})["results"]

def get_movie_details(movie_id):
    return tmdb_get(f"/movie/{movie_id}")

def get_recommendations(movie_id, page=1):
    return tmdb_get(f"/movie/{movie_id}/recommendations", {"page": page})["results"]

def get_movie_reviews(movie_id, page=1):
    return tmdb_get(f"/movie/{movie_id}/reviews", {"page": page})["results"]

def get_watch_providers(movie_id, country="US"):
    data = tmdb_get(f"/movie/{movie_id}/watch/providers")
    return data.get("results", {}).get(country, {})

def discover_movies(year_from=None, year_to=None, primary_release_year=None, vote_avg_min=None, language=None, page=1):
    params = {"page": page, "sort_by": "popularity.desc"}
    if vote_avg_min:
        params["vote_average.gte"] = vote_avg_min
    if year_from and year_to:
        params["primary_release_date.gte"] = f"{year_from}-01-01"
        params["primary_release_date.lte"] = f"{year_to}-12-31"
    if language:
        params["with_original_language"] = language
    return tmdb_get("/discover/movie", params)["results"]

# ====== WATCHLIST PERSISTENCE ======
def load_watchlists():
    if not os.path.exists(WATCHLIST_FILE):
        return {}
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_watchlists(data):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def add_to_watchlist(username, movie):
    lists = load_watchlists()
    user_list = lists.get(username, [])
    # Avoid duplicates by id
    if not any(m["id"] == movie["id"] for m in user_list):
        user_list.append({"id": movie["id"], "title": movie["title"], "poster_path": movie.get("poster_path")})
    lists[username] = user_list
    save_watchlists(lists)

def remove_from_watchlist(username, movie_id):
    lists = load_watchlists()
    user_list = lists.get(username, [])
    user_list = [m for m in user_list if m["id"] != movie_id]
    lists[username] = user_list
    save_watchlists(lists)

# ====== SENTIMENT ANALYSIS ======
sia = SentimentIntensityAnalyzer()
def average_review_sentiment(movie_id, n_pages=2):
    # fetch up to n_pages of reviews and score them
    sentiments = []
    for p in range(1, n_pages + 1):
        try:
            data = get_movie_reviews(movie_id, page=p)
        except:
            break
        for r in data.get("results", []):
            text = r.get("content", "")
            if text.strip():
                s = sia.polarity_scores(text)
                sentiments.append(s["compound"])
        if len(data.get("results", [])) == 0:
            break
    if not sentiments:
        return None
    return sum(sentiments) / len(sentiments)

# ====== CONTENT-BASED SIMILARITY (TF-IDF on overviews)
def content_similarity_recs(movie_id, candidate_movies, top_k=10):
    # candidate_movies: list of movie dicts with 'id' and 'overview'
    df = pd.DataFrame(candidate_movies)
    df["overview"] = df["overview"].fillna("")
    if df.shape[0] < 2:
        return []
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    # locate index of movie_id
    try:
        idx = df.index[df["id"] == movie_id][0]
    except IndexError:
        return []
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    # create series of (id, score), exclude self
    pairs = [(int(df.loc[i, "id"]), cosine_sim[i]) for i in df.index if df.loc[i, "id"] != movie_id]
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top = [pid for pid, score in pairs_sorted[:top_k]]
    # return movie dicts in original order
    return [df[df["id"] == tid].iloc[0].to_dict() for tid in top]

# ====== HYBRID RECOMMENDER ======
def hybrid_recommendations(base_movie_id, n_recs=10):
    # 1) TMDB recommendations (popularity-based)
    try:
        tmdb_recs = get_recommendations(base_movie_id, page=1)
    except:
        tmdb_recs = []
    # 2) Gather candidate movies for content similarity:
    #     use TMDB recs + discover popular movies as pool
    pool = tmdb_recs.copy()
    try:
        discovered = discover_movies(vote_avg_min=6, page=1)
        pool.extend(discovered[:40])  # small pool
    except:
        pass
    # deduplicate by id
    seen = {}
    candidates = []
    for m in pool:
        mid = m.get("id")
        if mid and mid not in seen:
            seen[mid] = True
            candidates.append(m)
    # content-based top matches
    content_recs = content_similarity_recs(base_movie_id, candidates, top_k=n_recs)
    # Merge scores: give TMDB recs weight 0.6, content weight 0.4
    # Score from TMDB recs: rank-based
    score_map = {}
    for rank, m in enumerate(tmdb_recs[:50]):
        score_map[m["id"]] = score_map.get(m["id"], 0) + 0.6 * (1.0 / (1 + rank))
    for rank, m in enumerate(content_recs):
        score_map[m["id"]] = score_map.get(m["id"], 0) + 0.4 * (1.0 / (1 + rank))
    # produce sorted list
    scored = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rec_ids = [mid for mid, _ in scored][:n_recs]
    final = []
    for rid in rec_ids:
        # fetch movie details (from tmdb_recs, candidates, or API fallback)
        found = next((m for m in candidates if m.get("id") == rid), None)
        if not found:
            try:
                found = get_movie_details(rid)
            except:
                continue
        final.append(found)
    return final

# ====== STREAMLIT UI ======
# Initialize session state variables if they don't exist
if "username" not in st.session_state:
    st.session_state["username"] = "guest"
if "selected_movie_id" not in st.session_state:
    st.session_state["selected_movie_id"] = None
if "discover_results" not in st.session_state:
    st.session_state["discover_results"] = None

st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Movie Recommendation System (Upgraded)")

# sidebar: user / filters
with st.sidebar:
    st.header("User & Settings")
    username = st.text_input("Enter username (for watchlist)", value=st.session_state.get("username", "guest"))
    st.session_state["username"] = username

    st.markdown("---")
    st.subheader("Advanced Filters")
    year_from, year_to = st.slider("Release year range", 1950, 2025, (2000, 2022))
    min_rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 6.0, 0.1)
    language = st.selectbox("Language (original)", ["", "en", "fr", "es", "de", "ja", "ko"])
    provider_country = st.selectbox("Provider Country (for availability)", ["US", "GB", "NG", "DE", "FR", "IN"], index=0)
    st.markdown("Note: provider availability depends on TMDB data.")

st.write("Search a movie to start or pick a trending/discover option.")
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Search movie by name", value="Inception")
    if st.button("Search"):
        results = search_movie(query)
        if not results:
            st.warning("No results found")
        else:
            # show first 10 results
            for movie in results[:10]:
                cols = st.columns([1, 4, 1])
                poster = movie.get("poster_path")
                with cols[0]:
                    if poster:
                        st.image(f"{POSTER_BASE}{poster}", width=80)
                with cols[1]:
                    st.markdown(f"**{movie.get('title')}** ({movie.get('release_date','')[:4]})")
                    st.caption(f"TMDB rating: {movie.get('vote_average', 'N/A')}")
                    st.write(movie.get("overview", "")[:300] + ("..." if len(movie.get("overview","")) > 300 else ""))
                    if st.button(f"View / Recommend - {movie['id']}", key=f"view_{movie['id']}"):
                        st.session_state["selected_movie_id"] = movie["id"]

with col2:
    st.subheader("Quick Discover")
    if st.button("Popular (Discover)"):
        try:
            disc = discover_movies(year_from, year_to, vote_avg_min=min_rating)
            st.session_state["discover_results"] = disc[:10]
        except Exception as e:
            st.error(f"Discover failed: {e}")
    if st.session_state.get("discover_results"):
        for m in st.session_state["discover_results"]:
            if st.button(f"Select: {m['title']}", key=f"select_{m['id']}"):
                st.session_state["selected_movie_id"] = m["id"]

# Display selected movie and recommendations
selected_id = st.session_state.get("selected_movie_id", None)
if selected_id:
    try:
        details = get_movie_details(selected_id)
    except Exception as e:
        st.error(f"Failed to fetch details: {e}")
        details = None

    if details:
        st.subheader(f"{details['title']} ({details.get('release_date','')[:4]})")
        left, right = st.columns([1, 2])
        with left:
            if details.get("poster_path"):
                st.image(f"{POSTER_BASE}{details['poster_path']}", width=200)
            st.markdown(f"**TMDB Rating:** {details.get('vote_average','N/A')}/10")
            providers = get_watch_providers(selected_id, country=provider_country)
            if providers:
                st.markdown("**Providers**")
                for k, v in providers.items():
                    st.write(f"{k}: {v}")

            # watchlist buttons
            if st.button("Add to Watchlist"):
                add_to_watchlist(username, details)
                st.success("Added to watchlist.")
            if st.button("Remove from Watchlist"):
                remove_from_watchlist(username, selected_id)
                st.info("Removed from watchlist (if present).")

        with right:
            st.markdown("**Overview**")
            st.write(details.get("overview", "No overview available."))
            # Sentiment
            with st.spinner("Fetching & analyzing reviews..."):
                sentiment_avg = average_review_sentiment(selected_id, n_pages=2)
                if sentiment_avg is None:
                    st.write("No review sentiment data available.")
                else:
                    st.write(f"Average review sentiment (compound): {sentiment_avg:.3f}")
                    if sentiment_avg >= 0.3:
                        st.success("Generally positive reviews")
                    elif sentiment_avg <= -0.2:
                        st.error("Generally negative reviews")
                    else:
                        st.info("Mixed / neutral reviews")

        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations (Hybrid)")
        with st.spinner("Building hybrid recommendations..."):
            recs = hybrid_recommendations(selected_id, n_recs=8)
        if not recs:
            st.info("No recommendations found.")
        else:
            for r in recs:
                cols = st.columns([1, 4])
                with cols[0]:
                    if r.get("poster_path"):
                        st.image(f"{POSTER_BASE}{r['poster_path']}", width=90)
                with cols[1]:
                    st.markdown(f"**{r.get('title')}** ({r.get('release_date','')[:4]})")
                    st.caption(f"TMDB rating: {r.get('vote_average','N/A')}")
                    st.write((r.get("overview") or "")[:300] + ("..." if len((r.get("overview") or "")) > 300 else ""))
                    # sentiment snippet
                    s = average_review_sentiment(r.get("id"), n_pages=1)
                    if s is not None:
                        st.write(f"Review sentiment (sample): {s:.2f}")

# Watchlist display
st.sidebar.markdown("---")
st.sidebar.subheader(f"{username}'s Watchlist")
lists = load_watchlists()
user_list = lists.get(username, [])
if user_list:
    for m in user_list:
        cols = st.sidebar.columns([1, 3])
        with cols[0]:
            if m.get("poster_path"):
                st.image(f"{POSTER_BASE}{m['poster_path']}", width=60)
        with cols[1]:
            st.sidebar.write(m["title"])
            if st.sidebar.button(f"Remove {m['id']}", key=f"rm_{m['id']}"):
                remove_from_watchlist(username, m["id"])
                st.experimental_rerun()

st.markdown("---")
st.caption("This app uses TMDB live data. For best results add your TMDB_API_KEY in the code.")

