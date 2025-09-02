import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
best_rf_model = pickle.load(open("optimized_random_forest_model.pkl", "rb"))
le_platform = pickle.load(open("platform_encoder.pkl", "rb"))
le_target = pickle.load(open("engagement_encoder.pkl", "rb"))

st.title("📊 Engagement Level Prediction")

# Input fields
platform = st.selectbox("Platform", le_platform.classes_)
views = st.number_input("Views", min_value=0)
likes = st.number_input("Likes", min_value=0)
shares = st.number_input("Shares", min_value=0)
comments = st.number_input("Comments", min_value=0)

if st.button("🔍 Predict Engagement"):
    # Encode and prepare input
    platform_encoded = le_platform.transform([platform])[0]
    input_data = [[platform_encoded, views, likes, shares, comments]]
    input_df = pd.DataFrame(input_data, columns=['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments'])
    
    # Predict
    prediction = best_rf_model.predict(input_df)[0]
    engagement_level = le_target.inverse_transform([prediction])[0]
    
    st.success(f"✅ Predicted Engagement Level: {engagement_level}")
