from flask import Flask, render_template, request, jsonify
import json
import os
from collections import Counter

app = Flask(__name__)

# Load the dataset at app start
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init.json')

with open(DATASET_PATH, 'r') as file:
    comments_data = json.load(file)

def preprocess_text(text):
    # Basic text preprocessing to lowercase and split by spaces
    return text.lower().split()

def keyword_in_subreddit(keywords, subreddit_text):
    # Count how many input keywords appear in the subreddit's text
    return sum(subreddit_text.count(keyword) for keyword in keywords)

@app.route("/")
def home():
    return render_template('base.html')

@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    query_keywords = preprocess_text(query)

    # Aggregate comments by subreddit
    subreddit_agg_texts = {}
    for comment in comments_data:
        subreddit = comment['subreddit']
        text = preprocess_text(comment['text'])
        if subreddit in subreddit_agg_texts:
            subreddit_agg_texts[subreddit] += text
        else:
            subreddit_agg_texts[subreddit] = text

    # Count occurrences of input words in each subreddit
    keyword_counts = {subreddit: keyword_in_subreddit(query_keywords, text)
                      for subreddit, text in subreddit_agg_texts.items()}

    # Get the top 3 subreddits based on keyword occurrences
    top_subreddits = sorted(keyword_counts, key=keyword_counts.get, reverse=True)[:3]

    return jsonify(top_subreddits)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)