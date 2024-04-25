from flask import Flask, render_template, request, jsonify
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from sklearn.decomposition import TruncatedSVD

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

app = Flask(__name__)

# Load the dataset at app start
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init.json")
with open(DATASET_PATH, "r") as file:
    comments_data = json.load(file)

# Aggregating texts by subreddit
subreddit_agg_texts = {}
for comment in comments_data:
    subreddit = comment["subreddit"]
    text = comment["text"].lower()
    if subreddit in subreddit_agg_texts:
        subreddit_agg_texts[subreddit] += " " + text
    else:
        subreddit_agg_texts[subreddit] = text

# Prepare data for TF-IDF
subreddits = list(subreddit_agg_texts.keys())
texts = list(subreddit_agg_texts.values())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Perform SVD
svd = TruncatedSVD(n_components=100)
svd_matrix = svd.fit_transform(tfidf_matrix)

def custom_stopwords():
    stop_words = set(stopwords.words("english"))
    more_stopwords = {
        "love",
        "like",
        "just",
        "also",
        "really",
        "very",
        "much",
        "can",
        "will",
        "one",
        "use",
        "would",
        "and",
        "or",
    }
    stop_words.update(more_stopwords)
    stop_words.update(string.ascii_lowercase)  # Add single letters
    return stop_words


def extract_keywords(text):
    stop_words = custom_stopwords()
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    keywords = [
        word
        for word, tag in tagged_words
        if tag.startswith("NN") and word.lower() not in stop_words
    ]
    return keywords


@app.route("/")
def home():
    return render_template("base.html")


@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    keywords = extract_keywords(query)
    if not keywords:
        return jsonify([])

    keyword_query = " ".join(keywords)
    query_vector = vectorizer.transform([keyword_query.lower()])
    query_svd = svd.transform(query_vector)

    # Calculate cosine similarities using SVD matrix
    cos_similarities = cosine_similarity(query_svd, svd_matrix)

    top_indices = np.argsort(cos_similarities[0])[::-1][:5]
    top_subreddits = [subreddits[index] for index in top_indices]
    similarity_scores = [cos_similarities[0][index] for index in top_indices]
    results = [
        {"subreddit": subreddit, "similarity_score": score}
        for subreddit, score in zip(top_subreddits, similarity_scores)
    ]
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)