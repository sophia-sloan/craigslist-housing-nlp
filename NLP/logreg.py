"""
TF-IDF + Logistic Regression classifier for Craigslist housing types.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import time

DATA_PATH = "../Data/housing.csv"

def load_data(path):
    df = pd.read_csv(path, usecols=["description", "type"])
    df = df.dropna(subset=["description", "type"])
    # Keep only types with enough samples to stratify on
    min_samples = 50
    counts = df["type"].value_counts()
    valid_types = counts[counts >= min_samples].index
    df = df[df["type"].isin(valid_types)]
    return df

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {len(df):,} listings | {df['type'].nunique()} housing types")
    print(f"  Type distribution:\n{df['type'].value_counts().to_string()}\n")

    X = df["description"].astype(str)
    y = df["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=5,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        )),
    ])

    print("Training TF-IDF + Logistic Regression pipeline...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s\n")

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Show the top TF-IDF features per class
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    print("\nTop 10 TF-IDF features per class:")
    for i, cls in enumerate(clf.classes_):
        top_idx = clf.coef_[i].argsort()[-10:][::-1]
        top_features = ", ".join(feature_names[top_idx])
        print(f"  {cls:20s}: {top_features}")

if __name__ == "__main__":
    main()
