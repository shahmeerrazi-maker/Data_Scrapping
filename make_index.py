import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load scraped CSV
csv_file = "scrapping_results.csv"
df = pd.read_csv(csv_file)

# Fill NaN values
df = df.fillna("")

# Combine title + text for indexing (use lowercase column names)
documents = (df["title"] + " " + df["text"]).tolist()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Learn vocabulary + transform all documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Save the index to disk
with open("index.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "titles": df["title"].tolist(),
        "urls": df["url"].tolist()
    }, f)

print("Index created successfully! Saved as index.pkl")
