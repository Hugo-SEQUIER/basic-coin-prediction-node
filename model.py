import json
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from config import data_base_path, model_file_path, NEWS_API_KEY
import requests
from datetime import datetime, timedelta

news_data_path = os.path.join(data_base_path, "news_data.json")
training_data_path = os.path.join(data_base_path, "training_data.csv")

def get_news_api_url(days=30):
    base_url = "https://newsapi.org/v2/everything"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "q": "US presidential election",
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }
    
    return base_url, params

def download_news_data(days=30):
    url, params = get_news_api_url(days)
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        articles = response.json()["articles"]
        with open(news_data_path, 'w') as f:
            json.dump(articles, f)
        print(f"Downloaded {len(articles)} news articles")
        return articles
    else:
        print(f"Failed to download news data: {response.status_code}")
        return None

def prepare_training_data():
    with open(news_data_path, 'r') as f:
        articles = json.load(f)
    
    df = pd.DataFrame(articles)
    df['text'] = df['title'] + " " + df['description'].fillna("")
    
    # This is a placeholder. In a real scenario, you'd need actual labeled data.
    # For demonstration, we're randomly assigning labels.
    df['label'] = pd.np.random.choice([0, 1], size=len(df))
    
    df[['text', 'label']].to_csv(training_data_path, index=False)
    print(f"Prepared training data with {len(df)} samples")

def train_model():
    df = pd.read_csv(training_data_path)
    X = df['text']
    y = df['label']

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression()),
    ])

    model.fit(X, y)

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_inference():
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    recent_articles = download_news_data(days=1)
    if recent_articles:
        texts = [article['title'] + " " + article.get('description', "") for article in recent_articles]
        probabilities = loaded_model.predict_proba(texts)
        republican_likelihood = probabilities[:, 1].mean()
        return republican_likelihood
    else:
        return None