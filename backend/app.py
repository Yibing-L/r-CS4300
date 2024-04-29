from flask import Flask, render_template, request, jsonify
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__, static_folder='static', template_folder='templates')

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init.json")
with open(DATASET_PATH, "r") as file:
    analyzed_data = json.load(file)

# extract subreddits and keyword strings
subreddits = list(analyzed_data.keys())
keyword_strings = [' '.join(keywords) for keywords in analyzed_data.values()]

vectorizer = TfidfVectorizer()

# fit and transform keyword strings to get TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(keyword_strings)

# do SVD on TF-IDF matrix
svd = TruncatedSVD(n_components=1000)
svd_matrix = svd.fit_transform(tfidf_matrix)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    query_vector = vectorizer.transform([query.lower()])
    query_svd = svd.transform(query_vector)

    # cosine similarities using SVD matrix
    cos_similarities = cosine_similarity(query_svd, svd_matrix)
    top_indices = np.argsort(cos_similarities[0])[::-1][:5]
    top_subreddits = [subreddits[index] for index in top_indices]
    similarity_scores = [cos_similarities[0][index] for index in top_indices]

    results = []
    for subreddit, score in zip(top_subreddits, similarity_scores):
        top_keywords = analyzed_data[subreddit][:10]
        top_keywords_str = (',' + ' ').join(keyword for keyword in top_keywords)

        result = {
            "subreddit": subreddit,
            "similarity_score": score,
            "top_keywords": top_keywords_str
        }
        results.append(result)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)