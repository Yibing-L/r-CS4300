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

# Compute TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

def custom_stopwords():
    # Custom stopwords to exclude more common but less informative words
    stop_words = set(stopwords.words('english'))
    more_stopwords = {'love', 'like', 'just', 'also', 'really', 'very', 'much', 'can', 'will', 'one', 'use', 'would','and','or'}
    stop_words.update(more_stopwords)
    stop_words.update(string.ascii_lowercase)  # Add single letters
    return stop_words

def extract_keywords(text):
    stop_words = custom_stopwords()
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    keywords = [word for word, tag in tagged_words if tag.startswith('NN') and word.lower() not in stop_words]
    return keywords

@app.route("/")
def home():
    return render_template('base.html')

@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    # Extract keywords using the enhanced method
    keywords = extract_keywords(query)

    # If no keywords, return empty
    if not keywords:
        return jsonify([])

    # Join keywords for vectorization
    keyword_query = ' '.join(keywords)
    query_vector = vectorizer.transform([keyword_query.lower()])

    # Compute cosine similarity
    cos_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Get the top 3 subreddits based on cosine similarity
    top_indices = np.argsort(cos_similarities[0])[::-1][:3]
    top_subreddits = [subreddits[index] for index in top_indices]

    return jsonify(top_subreddits)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)