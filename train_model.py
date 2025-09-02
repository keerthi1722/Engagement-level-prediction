import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pickle

# Load dataset
df = pd.read_csv("viral_social_media_trends_1000.csv")

# Encode categorical columns
le_platform = LabelEncoder()
df['Platform_Encoded'] = le_platform.fit_transform(df['Platform'])

le_target = LabelEncoder()
df['Engagement_Level_Encoded'] = le_target.fit_transform(df['Engagement_Level'])

# Features and target
features = ['Platform_Encoded', 'Views', 'Likes', 'Shares', 'Comments']
X = df[features]
y = df['Engagement_Level_Encoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train with GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Print a sample tree
tree_text = export_text(best_rf_model.estimators_[0], feature_names=features)
print("\n🌲 First Decision Tree:\n")
print(tree_text)

# Evaluate
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {round(accuracy*100, 2)} %")

# Save model and encoders
pickle.dump(best_rf_model, open("optimized_random_forest_model.pkl", "wb"))
pickle.dump(le_platform, open("platform_encoder.pkl", "wb"))
pickle.dump(le_target, open("engagement_encoder.pkl", "wb"))
print("✅ Model and encoders saved!")
