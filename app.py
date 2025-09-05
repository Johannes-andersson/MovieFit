# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Fit Checker", page_icon="🎬", layout="wide")
st.title("🎬 Is this movie a fit for me? (k-NN content-based)")

CSV_PATH = "movies_1989_2025.csv"   # your file name
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]
required = {'title', 'year', 'genres', 'runtime_min', 'my_rating'}
assert required.issubset(df.columns), f"CSV must have columns: {sorted(required)}"

# genres as lists + vocabulary
df['genres'] = df['genres'].astype(str).str.replace(", ", ",", regex=False).str.split(",")
KNOWN_GENRES = sorted({g for row in df['genres'] for g in row if g})

# ---------- Build feature space ----------
def build_features(df_):
    mlb = MultiLabelBinarizer()
    G = mlb.fit_transform(df_['genres'])
    genre_classes = list(mlb.classes_)
    num = df_[['runtime_min', 'year']].copy()
    num = num.fillna(num.median(numeric_only=True))
    scaler = StandardScaler().fit(num)
    num_scaled = scaler.transform(num)
    X = np.hstack([G, num_scaled])
    return X, genre_classes, scaler

X, GENRES_KNOWN, scaler = build_features(df)
knn_all = NearestNeighbors(n_neighbors=25, metric='cosine').fit(X)

# ---------- Helpers ----------
def encode_candidate(genres_list, runtime_min, year):
    gvec = np.zeros(len(GENRES_KNOWN), dtype=float)
    for g in (genres_list or []):
        for i, known in enumerate(GENRES_KNOWN):
            if g == known: gvec[i] = 1.0
    num = pd.DataFrame([[runtime_min, year]], columns=['runtime_min','year'])
    num = num.fillna(df[['runtime_min','year']].median(numeric_only=True))
    num_scaled = scaler.transform(num).ravel()
    return np.hstack([gvec, num_scaled])

def fit_score_vs_likes(vec, liked_mask, topn_neighbors=30):
    if liked_mask.sum() == 0:
        return np.nan
    X_liked = X[liked_mask]
    k = min(topn_neighbors, X_liked.shape[0])
    knn_liked = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X_liked)
    dist, _ = knn_liked.kneighbors(vec.reshape(1, -1))
    sims = 1 - dist[0]
    return float(np.mean(sims))

def omdb_lookup(title, apikey):
    r = requests.get("https://www.omdbapi.com/", params={"t": title, "apikey": apikey}, timeout=10)
    data = r.json()
    if data.get("Response") != "True":
        raise ValueError(data.get("Error", "Not found"))
    yr = data.get("Year")
    yr = int(str(yr).split("–")[0]) if yr else None
    rt = data.get("Runtime", "0 min").split()[0]
    rt = int(rt) if rt.isdigit() else None
    g  = [g.strip() for g in data.get("Genre","").split(",") if g.strip()]
    return yr, rt, g

# ---------- Sidebar ----------
st.sidebar.header("Settings")
like_threshold = st.sidebar.slider("Treat my rating ≥ as 'liked'", 3, 5, 4, 1)
k_neighbors = st.sidebar.slider("Neighbors to show (explanation)", 5, 30, 10, 1)
liked_mask = df['my_rating'] >= like_threshold

# ---------- Versioned key pattern (prevents 'cannot be modified' error) ----------
if "ver" not in st.session_state:
    st.session_state.ver = 0
if "prefill" not in st.session_state:
    st.session_state.prefill = {
        "title": "",
        "year": 1995,
        "runtime": 110,
        "genres": []
    }

ver = st.session_state.ver  # used to make unique keys per render

# Use prefill values as the *current* values for widgets.
st.subheader("Describe a movie (not necessarily in your CSV)")
title = st.text_input("Title (free text)",
                      value=st.session_state.prefill["title"],
                      key=f"title_{ver}")

c1, c2, c3 = st.columns(3)
with c1:
    year = st.number_input("Year", min_value=1900, max_value=2100,
                           value=int(st.session_state.prefill["year"]),
                           key=f"year_{ver}")
with c2:
    runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=400,
                              value=int(st.session_state.prefill["runtime"]),
                              key=f"runtime_{ver}")
with c3:
    genres = st.multiselect("Genres (multi-select, from your library)",
                            options=KNOWN_GENRES,
                            default=st.session_state.prefill["genres"],
                            key=f"genres_{ver}")

# ---------- OMDb auto-fill ----------
st.markdown("**Auto-fill from OMDb (optional)**")
omdb_key = st.text_input("OMDb API key", type="password", help="Only the raw key (e.g., a7b9c2d4)")

if st.button("Auto-fill from OMDb"):
    if not omdb_key:
        st.warning("Enter your OMDb API key first.")
    elif not title.strip():
        st.warning("Enter a movie title first.")
    else:
        try:
            yr, rt, g = omdb_lookup(title.strip(), omdb_key)
            g_clean = [x for x in (g or []) if x in KNOWN_GENRES]
            # update prefill values and bump version to rebuild widgets with new defaults
            st.session_state.prefill = {
                "title": title.strip(),
                "year": yr if yr is not None else year,
                "runtime": rt if rt is not None else runtime,
                "genres": g_clean if g_clean else genres
            }
            st.session_state.ver += 1
            st.rerun()
        except Exception as e:
            st.error(f"OMDb error: {e}")

# ---------- Check fit ----------
if st.button("Check fit"):
    # read the CURRENT values (not prefill)
    cur_title  = st.session_state.get(f"title_{ver}", title)
    cur_year   = st.session_state.get(f"year_{ver}", year)
    cur_runtime= st.session_state.get(f"runtime_{ver}", runtime)
    cur_genres = st.session_state.get(f"genres_{ver}", genres)

    if not cur_genres:
        st.warning("Pick at least one genre (or use Auto-fill).")
    else:
        vec = encode_candidate(cur_genres, cur_runtime, cur_year)
        score = fit_score_vs_likes(vec, liked_mask)
        if np.isnan(score):
            st.error("You have no 'liked' movies yet. Increase ratings in CSV or lower the threshold.")
        else:
            st.subheader("✅ Fit Score")
            st.metric("Average similarity to your liked movies", f"{100*score:.1f}%")
            st.caption("Computed via cosine similarity to your liked set.")

        k = min(k_neighbors, X.shape[0])
        dist_all, idx_all = knn_all.kneighbors(vec.reshape(1, -1), n_neighbors=k)
        sims_all = 1 - dist_all[0]
        recs = (df.iloc[idx_all[0]].copy()
                  .assign(similarity=np.round(sims_all, 3))
                  [['title','year','genres','runtime_min','my_rating','similarity']])
        st.subheader("Closest movies in your library (explanation)")
        st.dataframe(recs.reset_index(drop=True))

st.caption("If Auto-fill still shows the old values, it’s cached by the browser. Click Auto-fill again or change the title slightly.")

