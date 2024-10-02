from flask import Flask, Response
import requests
import json
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta
import schedule
import time

# create our Flask app
app = Flask(__name__)

# define the Hugging Face model we will use
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

def get_news_api_url():
    base_url = "https://newsapi.org/v2/everything"
    # Set the date range for the last 24 hours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    params = {
        "q": "US presidential election",
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": "<Your News API key>"  # replace with your API key
    }
    
    return base_url, params

@app.route("/inference")
def get_inference():
    """Generate inference for US presidential election."""
    try:
        # Get news articles
        url, params = get_news_api_url()
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            articles = response.json()["articles"]
            # Analyze sentiment of article titles
            sentiments = [sentiment_pipeline(article["title"])[0] for article in articles]
            
            # Calculate the likelihood of Republican win based on sentiment
            positive_sentiments = sum(1 for s in sentiments if s["label"] == "POSITIVE")
            likelihood = positive_sentiments / len(sentiments)
            
            # Adjust likelihood (this is a simplification and should be refined)
            republican_likelihood = likelihood if likelihood > 0.5 else 1 - likelihood
            
            return Response(str(republican_likelihood), status=200)
        else:
            return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}), 
                            status=response.status_code, 
                            mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

def daily_inference():
    """Function to run inference daily and print the result."""
    with app.app_context():
        result = get_inference()
        print(f"Daily inference result: {result.get_data(as_text=True)}")

# Schedule the daily inference
schedule.every().day.at("00:00").do(daily_inference)

# run our Flask app
if __name__ == '__main__':
    # Run the scheduler in a separate thread
    import threading
    scheduler_thread = threading.Thread(target=lambda: schedule.run_pending())
    scheduler_thread.start()
    
    app.run(host="0.0.0.0", port=8060, debug=True)