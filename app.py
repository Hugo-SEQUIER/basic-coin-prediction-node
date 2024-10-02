from flask import Flask, Response
import json
import schedule
import time
from model import download_news_data, prepare_training_data, train_model, get_inference

# create our Flask app
app = Flask(__name__)

@app.route("/inference/<token>")
def inference(token):
    """Generate inference for US presidential election."""
    try:
        republican_likelihood = get_inference()
        if republican_likelihood is not None:
            return Response(json.dumps({"value": str(republican_likelihood)}), status=200, mimetype='application/json')
        else:
            return Response(json.dumps({"error": "Failed to generate inference"}), 
                            status=500, 
                            mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update the model with new data."""
    try:
        download_news_data()
        prepare_training_data()
        train_model()
        return Response("0", status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

def daily_update():
    """Function to run update daily."""
    with app.app_context():
        result = update()
        print(f"Daily update result: {result.get_data(as_text=True)}")

# Schedule the daily update
schedule.every().day.at("00:00").do(daily_update)

# run our Flask app
if __name__ == '__main__':
    # Run the scheduler in a separate thread
    import threading
    def run_scheduler():
        while True:
            schedule.run_pending()

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()

    app.run(host="0.0.0.0", port=8060, debug=True)