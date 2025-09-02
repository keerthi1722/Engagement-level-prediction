from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Engagement Level Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; background: #f5f5f5; }
        .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; }
        input, select, button { width: 100%; padding: 10px; margin: 8px 0; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #007bff; color: white; font-size: 16px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; text-align: center; font-size: 20px; font-weight: bold; color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Engagement Level Prediction</h2>
        <form method="POST" action="/predict">
            <label>Platform:</label>
            <select name="Platform">
                <option value="YouTube">YouTube</option>
                <option value="Instagram">Instagram</option>
                <option value="Twitter">Twitter</option>
                <option value="Facebook">Facebook</option>
            </select>
            <label>Views:</label>
            <input type="number" name="Views" required>
            <label>Likes:</label>
            <input type="number" name="Likes" required>
            <label>Shares:</label>
            <input type="number" name="Shares" required>
            <label>Comments:</label>
            <input type="number" name="Comments" required>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
        <div class="result">Predicted Engagement Level: {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Platform": [request.form["Platform"]],
        "Views": [int(request.form["Views"])],
        "Likes": [int(request.form["Likes"])],
        "Shares": [int(request.form["Shares"])],
        "Comments": [int(request.form["Comments"])]
    }
    df = pd.DataFrame(data)
    prediction = model.predict(df)[0]
    return render_template_string(HTML_PAGE, result=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
