import pandas as pd
import numpy as np
import re
import os
import html as html_lib
from collections import Counter
import math

# File paths
input_file = r"C:\Users\savin\OneDrive\Desktop\AI Ass\LAB 17\movie_reviews-1.csv"
output_file = os.path.join(os.path.dirname(input_file), "movie_updated.csv")

# Load data
df = pd.read_csv(input_file)

# Determine text column
if 'review' in df.columns:
    text_col = 'review'
elif 'review_text' in df.columns:
    text_col = 'review_text'
else:
    text_col = None

# --- BEFORE SUMMARY ---
print("=== BEFORE CLEANING ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing ratings:", df['rating'].isna().sum() if 'rating' in df.columns else "N/A")
if text_col is not None and len(df) > 0 and pd.notna(df[text_col].iloc[0]):
    print("Sample review (raw):", df[text_col].iloc[0])

# 1. Clean text
def clean_text(text):
    if pd.isnull(text):
        return ""
    raw = str(text)
    unescaped = html_lib.unescape(raw)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    normalized = re.sub(r"\s+", " ", no_tags).strip().lower()
    return normalized

if text_col is not None:
    df['review_clean'] = df[text_col].apply(clean_text)
else:
    raise ValueError("No 'review' or 'review_text' column found in the input CSV.")

# 2. Build simple TF-IDF manually
docs = df['review_clean'].tolist()
tokenized_docs = [re.findall(r"\b[a-z]+\b", doc) for doc in docs]

# Build vocabulary (top 100 frequent words, ignoring stopwords)
stopwords = set([
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "in", "on", 
    "of", "to", "for", "with", "as", "was", "were", "by", "at", "from"
])
word_counts = Counter([word for doc in tokenized_docs for word in doc if word not in stopwords])
vocab = [w for w, _ in word_counts.most_common(100)]

# Compute DF (document frequency)
df_counts = {word: sum(1 for doc in tokenized_docs if word in doc) for word in vocab}
N = len(tokenized_docs)

def compute_tfidf(doc):
    counts = Counter(doc)
    vec = []
    for word in vocab:
        tf = counts[word] / len(doc) if len(doc) > 0 else 0
        idf = math.log((N + 1) / (df_counts[word] + 1)) + 1
        vec.append(tf * idf)
    return vec

tfidf_matrix = np.array([compute_tfidf(doc) for doc in tokenized_docs])
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{w}" for w in vocab])
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# 3. Handle missing ratings
if 'rating' in df.columns:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    median_rating = df['rating'].median()
    df['rating_filled'] = df['rating'].fillna(median_rating)
else:
    raise ValueError("No 'rating' column found in the input CSV.")

# 4. Normalize ratings (0–10 → 0–1 scale)
df['rating_norm'] = df['rating_filled'] / 10.0

# --- AFTER SUMMARY ---
print("\n=== AFTER CLEANING ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing ratings (after fill):", df['rating_filled'].isna().sum())
print("Median rating used for fill:", median_rating)
if len(df) > 0 and 'review_clean' in df.columns and pd.notna(df['review_clean'].iloc[0]):
    print("Sample review (cleaned):", df['review_clean'].iloc[0])
print("TF-IDF features:", [col for col in df.columns if col.startswith("tfidf_")][:5], "...")

# Save cleaned dataset
try:
    df.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file}")
except PermissionError:
    alt_output = (
        output_file[:-4] + "_new.csv" if output_file.lower().endswith('.csv') else output_file + "_new"
    )
    df.to_csv(alt_output, index=False)
    print(f"\nOriginal file locked. Cleaned dataset saved to: {alt_output}")
