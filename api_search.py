from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load the index
# -------------------------
with open("index.pkl", "rb") as f:
    data = pickle.load(f)

vectorizer = data["vectorizer"]
tfidf_matrix = data["tfidf_matrix"]
titles = data["titles"]
urls = data["urls"]

# For simplicity, we assume claps are numeric; you can sort by claps if available
claps = [0] * len(titles)  # replace 0 with real claps if needed from CSV

# -------------------------
# Create Flask app
# -------------------------
app = Flask(__name__)

@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Input:
        JSON with either 'text' or 'keywords' key
    Output:
        top 10 similar articles: [{'title': ..., 'url': ...}, ...]
    """
    query = ""
    if request.method == "POST":
        req = request.get_json()
        query = req.get("text", "") or " ".join(req.get("keywords", []))
    else:
        query = request.args.get("text", "") or " ".join(request.args.getlist("keywords"))

    if not query:
        return jsonify({"error": "Provide 'text' or 'keywords' input"}), 400

    # Vectorize query
    q_vec = vectorizer.transform([query])

    # Compute cosine similarity
    sim = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Get top 10 indexes (highest similarity)
    top_idx = np.argsort(sim)[::-1][:10]

    # Build response
    results = []
    for i in top_idx:
        results.append({
            "title": titles[i],
            "url": urls[i],
            "similarity": float(sim[i])
        })

    return jsonify(results)

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5001, debug=True)

