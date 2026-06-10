"""
Sentence-BERT embedding classifier
Encodes listing descriptions with a pretrained SBERT model, then trains a
Logistic Regression on top of the fixed embeddings.

The full dataset is too large to encode in one pass, so a sample
of 20000 listings is used by default
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import time

DATA_PATH = "../Data/housing.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  
SAMPLE_SIZE = 20_000
BATCH_SIZE = 256
RANDOM_STATE = 42

def load_data(path, sample_size):
    df = pd.read_csv(path, usecols=["description", "type"])
    df = df.dropna(subset=["description", "type"])
    min_samples = 50
    counts = df["type"].value_counts()
    valid_types = counts[counts >= min_samples].index
    df = df[df["type"].isin(valid_types)]
    if sample_size and len(df) > sample_size:
        df = df.groupby("type", group_keys=False).apply(
            lambda g: g.sample(min(len(g), int(sample_size * len(g) / len(df))), random_state=RANDOM_STATE)
        )
        # top up to exactly sample_size
        if len(df) < sample_size:
            extra = df.sample(sample_size - len(df), random_state=RANDOM_STATE)
            df = pd.concat([df, extra]).drop_duplicates()
    return df.reset_index(drop=True)

def main():
    print(f"Loading data (stratified sample of {SAMPLE_SIZE:,})...")
    df = load_data(DATA_PATH, SAMPLE_SIZE)
    print(f"  {len(df):,} listings | {df['type'].nunique()} housing types")
    print(f"  Type distribution:\n{df['type'].value_counts().to_string()}\n")

    texts = df["description"].astype(str).tolist()

    le = LabelEncoder()
    y = le.fit_transform(df["type"])

    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts):,} descriptions (batch size {BATCH_SIZE})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Encoding time: {time.time() - t0:.1f}s")
    print(f"  Embedding matrix shape: {embeddings.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    print("Training Logistic Regression on SBERT embeddings...")
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s\n")

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

    # Nearest-neighbour examples: find the 3 training samples closest to each test centroid
    print("\nSample nearest-neighbour lookup per class (from test set):")
    test_descriptions = [texts[i] for i in range(len(texts)) if i >= int(0.8 * len(texts))]
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        centroid = X_test[mask].mean(axis=0)
        diffs = X_train - centroid
        dists = np.linalg.norm(diffs, axis=1)
        nearest = np.argmin(dists)
        # recover original text via index mapping
        train_indices = [i for i in range(len(texts)) if i < int(0.8 * len(texts))]
        nearest_text = texts[train_indices[nearest]] if nearest < len(train_indices) else "(unavailable)"
        print(f"\n  [{cls_name}] nearest training text snippet:")
        print(f"    {nearest_text[:200].strip()}...")

if __name__ == "__main__":
    main()
