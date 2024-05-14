from flask import Flask, render_template, request, jsonify
import json
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load data from JSON file
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init.json")
with open(DATASET_PATH, "r") as file:
    posts_data = json.load(file)

COMMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comments.json")
with open(COMMENTS_PATH, "r") as file:
    comments_data = json.load(file)

# Extract subreddits and their combined texts
subreddit_names = list(posts_data.keys())
keyword_strings = [' '.join(posts_data[subreddit]) for subreddit in subreddit_names]

custom_stop_words = stopwords.words('english') + [
    'https', 'com', 'www', 'https www', 'link', 'redd', 'amp', 'mt', 'gt',
    'reddit', 'imgur', 'jpg', 'wiki', 'html', 'http', 'net', 'org']

# Vectorize the keyword strings
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, min_df=1, max_df=0.99)
tfidf_matrix = vectorizer.fit_transform(keyword_strings)

# Perform SVD
svd = TruncatedSVD(n_components=9,random_state=2)  # Adjust number of components as needed
svd_matrix = svd.fit_transform(tfidf_matrix)

# Define category names for the SVD dimensions
category_names = tags = [
    "Society",
    "Politics",
    "Relationships",
    "Game",
    "Sports",
    "Entertainment",
    "Science",
    "Technology",
    "Economy"
]


@app.route("/")
def home():
    return render_template("base.html")

@app.route("/recommend", methods=["GET"])
def recommend_subreddits():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    # Transform the query and project it into the SVD space
    query_vector = vectorizer.transform([query.lower()])
    query_svd = svd.transform(query_vector).flatten()  # Flatten to make it 1D for dot product

    # Calculate dot products of the query vector with each subreddit's SVD vector
    dot_products = np.dot(svd_matrix, query_svd)

    # Filter results to only include non-zero similarity scores
    valid_indices = [i for i in range(len(dot_products)) if dot_products[i] > 0]
    if not valid_indices:
        return jsonify([])  # Return empty list if no valid indices (non-zero scores)

    # Sort the valid indices based on the dot products in descending order
    top_indices = sorted(valid_indices, key=lambda i: dot_products[i], reverse=True)[:5]  # Get indices of the top 5 results

    results = []
    for index in top_indices:
        subreddit_name = subreddit_names[index]
        # Calculate the element-wise product of the query vector and the subreddit's SVD vector
        elementwise_product = query_svd * svd_matrix[index]
        # Get indices of the top 3 dimensions based on the absolute values of the products
        top_component_indices = np.argsort(elementwise_product)[-3:][::-1]
        top_dimensions_str = ', '.join(category_names[i] for i in top_component_indices)

        # Find the most related comment for the subreddit

        related_comments = [comment for comment in comments_data if comment['subreddit'] == subreddit_name]
        if related_comments:
            # Vectorize the comments of the related subreddit
            related_comment_vectors = vectorizer.transform([comment['text'] for comment in related_comments])
            # Project these vectors into the same SVD space as the subreddit vectors
            related_comment_svd = svd.transform(related_comment_vectors)

            # Calculate the dot products of the query SVD with each comment's SVD vector
            dot_products = np.dot(related_comment_svd, query_svd)

            # Find the index of the comment with the highest dot product
            most_related_comment_idx = np.argmax(dot_products)
            most_related_comment = related_comments[most_related_comment_idx]['text']
        else:
            most_related_comment = "No comments available"


        result = {
            "subreddit": subreddit_name,
            "top_dimensions": top_dimensions_str,
            "related_comment": most_related_comment
        }
        results.append(result)

    return jsonify(results)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

