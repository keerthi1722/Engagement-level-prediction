import pickle
import pandas as pd
from flask import Flask, request, render_template

# Load model and encoders
model = pickle.load(open("optimized_random_forest_model.pkl", "rb"))
le_platform = pickle.load(open("platform_encoder.pkl", "rb"))
le_target = pickle.load(open("engagement_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            platform = request.form["platform"]
            views = int(request.form["views"])
            likes = int(request.form["likes"])
            shares = int(request.form["shares"])
            comments = int(request.form["comments"])

            if platform not in le_platform.classes_:
                prediction = f"❌ Platform '{platform}' not recognized!"
            else:
                platform_encoded = le_platform.transform([platform])[0]
                input_data = [platform_encoded, views, likes, shares, comments]
                input_df = pd.DataFrame([input_data], columns=['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments'])
                pred = model.predict(input_df)[0]
                prediction = le_target.inverse_transform([pred])[0]
        except Exception as e:
            prediction = f"❌ Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
