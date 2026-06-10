"""
BERTopic topic model

A sample of 15000 listings is used by default for speed
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import time

DATA_PATH = "../Data/housing.csv"
SBERT_MODEL = "all-MiniLM-L6-v2"
SAMPLE_SIZE = 15_000    #up to 385k
RANDOM_STATE = 42

# BERTopic hyperparameters
TOP_N_WORDS = 10        # keywords shown per topic
NR_TOPICS = "auto"     
MIN_TOPIC_SIZE = 50     # minimum docs per topic

def load_data(path, sample_size):
    df = pd.read_csv(path, usecols=["description", "type"])
    df = df.dropna(subset=["description", "type"])
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)

def main():
    print(f"Loading data (sample of {SAMPLE_SIZE:,})...")
    df = load_data(DATA_PATH, SAMPLE_SIZE)
    print(f"  {len(df):,} listings loaded\n")

    texts = df["description"].astype(str).tolist()

    # Encode with Sbert
    print(f"Encoding descriptions with SBERT ({SBERT_MODEL})...")
    sbert = SentenceTransformer(SBERT_MODEL)
    t0 = time.time()
    embeddings = sbert.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s | shape: {embeddings.shape}\n")

    # Configure Umap
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Bertopic
    print("Fitting BERTopic model...")
    topic_model = BERTopic(
        embedding_model=sbert,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=TOP_N_WORDS,
        nr_topics=NR_TOPICS,
        verbose=True,
    )
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts, embeddings)
    print(f"  BERTopic fit in {time.time() - t0:.1f}s\n")



    # Report
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (np.array(topics) == -1).sum()
    print(f"Discovered topics : {n_topics}")
    print(f"Outlier docs (-1) : {n_outliers} ({100 * n_outliers / len(topics):.1f}%)\n")

    print("=" * 70)
    print("Topic summary (excluding outlier cluster -1):")
    print("=" * 70)
    display_cols = ["Topic", "Count", "Name"]
    display_cols = [c for c in display_cols if c in topic_info.columns]
    print(topic_info[topic_info["Topic"] != -1][display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("Top keywords per topic:")
    print("=" * 70)
    for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]:
        words = topic_model.get_topic(topic_id)
        if words:
            kws = ", ".join([w for w, _ in words[:TOP_N_WORDS]])
            count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
            print(f"  Topic {topic_id:3d} ({count:5,} docs): {kws}")




    # Analysis
    print("\n" + "=" * 70)
    print("Topic distribution by housing type (top 5 topics per type):")
    print("=" * 70)
    df["topic"] = topics
    for housing_type, grp in df.groupby("type"):
        top_topics = (
            grp[grp["topic"] != -1]["topic"]
            .value_counts()
            .head(5)
        )
        if len(top_topics) == 0:
            continue
        top_str = ", ".join([f"T{t}({c})" for t, c in top_topics.items()])
        print(f"  {housing_type:20s}: {top_str}")

    # ── Step 6: show representative docs per topic ────────────────────────
    print("\n" + "=" * 70)
    print("Representative document snippet per topic (first 5 topics):")
    print("=" * 70)
    for topic_id in sorted(topic_info[topic_info["Topic"] != -1]["Topic"])[:5]:
        rep_docs = topic_model.get_representative_docs(topic_id)
        if rep_docs:
            snippet = rep_docs[0][:300].replace("\n", " ").strip()
            print(f"\n  [Topic {topic_id}]\n  {snippet}...")

if __name__ == "__main__":
    main()
