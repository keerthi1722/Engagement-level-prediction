from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("optimized_random_forest_model.pkl", "rb"))
le_platform = pickle.load(open("platform_encoder.pkl", "rb"))
le_target = pickle.load(open("engagement_encoder.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        platform = data['Platform']
        views = int(data['Views'])
        likes = int(data['Likes'])
        shares = int(data['Shares'])
        comments = int(data['Comments'])

        # Encode platform
        platform_encoded = le_platform.transform([platform])[0]

        # Prepare input
        input_data = [[platform_encoded, views, likes, shares, comments]]
        input_df = pd.DataFrame(input_data, columns=['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments'])

        # Predict
        prediction = model.predict(input_df)[0]
        engagement_level = le_target.inverse_transform([prediction])[0]

        return jsonify({"engagement_level": engagement_level})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
