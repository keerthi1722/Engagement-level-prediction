import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text  # Added for printing tree
import tkinter as tk
from tkinter import messagebox
import pickle

# Load the dataset
df = pd.read_csv(r'D:\\iml\\j3\\viral_social_media_trends_1000.csv')

# Encode the 'Platform' column
le_platform = LabelEncoder()
df['Platform_Encoded'] = le_platform.fit_transform(df['Platform'])

# Encode the target column 'Engagement_Level'
le_target = LabelEncoder()
df['Engagement_Level_Encoded'] = le_target.fit_transform(df['Engagement_Level'])

# Define features and target
features = ['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments']
X = df[features]
y = df['Engagement_Level_Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train Random Forest Classifier with Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# ✅ Print one of the trees in the random forest (e.g., the first one)
tree_text = export_text(best_rf_model.estimators_[0], feature_names=features)
print("\n🌲 First Decision Tree in the Random Forest:\n")
print(tree_text)

# Predict and evaluate
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Optimized Model Accuracy: {round(accuracy * 100, 2)} %")

# Save the optimized model and label encoders
pickle.dump(best_rf_model, open("optimized_random_forest_model.pkl", "wb"))
pickle.dump(le_platform, open("platform_encoder.pkl", "wb"))
pickle.dump(le_target, open("engagement_encoder.pkl", "wb"))

# -------------------- GUI APPLICATION --------------------

# Load the trained model and label encoders
best_rf_model = pickle.load(open("optimized_random_forest_model.pkl", "rb"))
le_platform = pickle.load(open("platform_encoder.pkl", "rb"))
le_target = pickle.load(open("engagement_encoder.pkl", "rb"))

# Tkinter UI
root = tk.Tk()
root.title("Engagement Level Prediction")
root.geometry("500x650")
root.configure(bg="#1E1E2F")

LABEL_COLOR = "#FFFFFF"
ENTRY_BG = "#2A2A40"
ENTRY_FG = "#FFFFFF"
BUTTON_BG = "#4CAF50"
BUTTON_FG = "#FFFFFF"
FONT = ("Arial", 12)

tk.Label(root, text="📊 Engagement Level Prediction", fg="#00ADB5", bg="#1E1E2F", font=("Arial", 16, "bold")).pack(pady=15)

fields = ["Platform", "Views", "Likes", "Shares", "Comments"]
entries = {}

for field in fields:
    tk.Label(root, text=f"{field}:", fg=LABEL_COLOR, bg="#1E1E2F", font=FONT).pack(anchor="w", padx=20, pady=2)
    entry = tk.Entry(root, font=FONT, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground="white", relief="flat")
    entry.pack(fill="x", padx=20, pady=5, ipady=5)
    entries[field] = entry

def predict_engagement():
    try:
        # Retrieve user input
        platform = entries["Platform"].get()
        views = int(entries["Views"].get())
        likes = int(entries["Likes"].get())
        shares = int(entries["Shares"].get())
        comments = int(entries["Comments"].get())

        # Edge case
        if views == 0 and likes == 0 and shares == 0 and comments == 0:
            engagement_level = 'Low'
            messagebox.showinfo("Prediction", f"✅ Predicted Engagement Level: {engagement_level}")
            return

        # Encode platform
        if platform not in le_platform.classes_:
            messagebox.showerror("Invalid Input", f"❌ Platform: '{platform}' is not recognized!")
            return
        platform_encoded = le_platform.transform([platform])[0]

        # Prepare input
        input_data = [platform_encoded, views, likes, shares, comments]
        input_df = pd.DataFrame([input_data], columns=['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments'])

        # Predict
        prediction = best_rf_model.predict(input_df)[0]
        engagement_level = le_target.inverse_transform([prediction])[0]

        messagebox.showinfo("Prediction", f"✅ Predicted Engagement Level: {engagement_level}")

    except ValueError:
        messagebox.showerror("Input Error", "❌ Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"❌ {str(e)}")

tk.Button(root, text="🔍 Predict Engagement", command=predict_engagement,
          font=FONT, bg=BUTTON_BG, fg=BUTTON_FG, relief="flat", padx=10, pady=10).pack(pady=25, ipadx=10, ipady=5)

root.mainloop()
