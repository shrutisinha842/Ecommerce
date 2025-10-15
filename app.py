from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Database configuration
app.secret_key = "secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model classes for the tables
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n + 1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details


# Routes
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def home():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))


@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=train_data, truncate=truncate)


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random.choice(random_image_urls),
                               random_price=random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]),
                               signup_message='User signed up successfully!')


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        # Verify user credentials
        user = Signin.query.filter_by(username=username, password=password).first()
        if user:
            return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                                   random_product_image_urls=random.choice(random_image_urls),
                                   random_price=random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]),
                                   signin_message='User signed in successfully!')
        else:
            return render_template('signin.html', error_message='Invalid username or password.')


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))

        # Debugging: Print to verify inputs and outputs
        print(f"Requested product: {prod}, Number of recommendations: {nbr}")

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        # Initialize variables to avoid UnboundLocalError
        random_product_image_urls = []
        random_prices = []
        message = ""

        # Check if recommendations are available
        if content_based_rec.empty:
            message = "No recommendations available for this product."
            print("No recommendations found.")
        else:
            # Generate image URLs and random prices
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            random_prices = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(nbr)]

        return render_template(
            'main.html',
            content_based_rec=content_based_rec,
            truncate=truncate,
            random_product_image_urls=random_product_image_urls,
            random_price=random_prices,
            message=message
        )


if __name__ == '__main__':
    app.run(debug=True)