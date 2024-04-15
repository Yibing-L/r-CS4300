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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Load the dataset at app start
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init.json')
with open(DATASET_PATH, 'r') as file:
    comments_data = json.load(file)

# Aggregating texts by subreddit
subreddit_agg_texts = {}
for comment in comments_data:
    subreddit = comment['subreddit']
    text = comment['text'].lower()
    if subreddit in subreddit_agg_texts:
        subreddit_agg_texts[subreddit] += " " + text
    else:
        subreddit_agg_texts[subreddit] = text

# Prepare data for TF-IDF
subreddits = list(subreddit_agg_texts.keys())
texts = list(subreddit_agg_texts.values())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Store user feedback
user_feedback = {}

def custom_stopwords():
    stop_words = set(stopwords.words('english'))
    more_stopwords = {'love', 'like', 'just', 'also', 'really', 'very', 'much', 'can', 'will', 'one', 'use', 'would', 'and', 'or'}
    stop_words.update(more_stopwords)
    stop_words.update(string.ascii_lowercase)  # Add single letters
    return stop_words

def extract_keywords(text):
    stop_words = custom_stopwords()
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    keywords = [word for word, tag in tagged_words if tag.startswith('NN') and word.lower() not in stop_words]
    return keywords

def rocchio(query_vector, relevant_vectors, irrelevant_vectors, alpha=1, beta=0.75, gamma=0.25):
    query_vector = np.asarray(query_vector.mean(axis=0)).ravel()
    
    if relevant_vectors is not None and relevant_vectors.shape[0] > 0:
        relevant_centroid = np.asarray(relevant_vectors.mean(axis=0)).ravel()
    else:
        relevant_centroid = np.zeros_like(query_vector)
    
    if irrelevant_vectors is not None and irrelevant_vectors.shape[0] > 0:
        irrelevant_centroid = np.asarray(irrelevant_vectors.mean(axis=0)).ravel()
    else:
        irrelevant_centroid = np.zeros_like(query_vector)

    new_query_vector = alpha * query_vector + beta * relevant_centroid - gamma * irrelevant_centroid
    return new_query_vector.reshape(1, -1)

@app.route("/")
def home():
    return render_template('base.html')

@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    keywords = extract_keywords(query)
    if not keywords:
        return jsonify([])

    keyword_query = ' '.join(keywords)
    query_vector = vectorizer.transform([keyword_query.lower()])

    # Get the user feedback for the current query
    query_feedback = user_feedback.get(query, {})

    relevant_indices = [i for i, s in enumerate(subreddits) if query_feedback.get(s, False)]
    irrelevant_indices = [i for i, s in enumerate(subreddits) if query_feedback.get(s, True) is False]

    relevant_vectors = tfidf_matrix[relevant_indices] if relevant_indices else None
    irrelevant_vectors = tfidf_matrix[irrelevant_indices] if irrelevant_indices else None

    if relevant_vectors is not None or irrelevant_vectors is not None:
        query_vector = rocchio(query_vector, relevant_vectors, irrelevant_vectors)

    cos_similarities = cosine_similarity(query_vector, tfidf_matrix)

    top_indices = np.argsort(cos_similarities[0])[::-1][:3]
    top_subreddits = [subreddits[index] for index in top_indices]

    return jsonify(top_subreddits)

@app.route("/feedback", methods=["POST"])
def feedback():
    query = request.args.get("query")
    subreddit = request.args.get("subreddit")
    is_relevant = request.args.get("isRelevant", 'false').lower() == 'true'
    query_feedback = user_feedback.get(query, {})
    query_feedback[subreddit] = is_relevant
    user_feedback[query] = query_feedback
    return "Feedback received", 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)