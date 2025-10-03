
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("churn_data.csv")

# Features & Target
target = "Churn"
features = [col for col in df.columns if col != target and col != "customerID"]

X = df[features]
y = df[target].apply(lambda x: 1 if x in ["Yes", 1] else 0)

# Identify categorical & numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 2. Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# 3. Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

report = classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

with open("evaluation_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nROC AUC: {roc:.4f}")

# 7. Feature importance (from RandomForest inside pipeline)
clf = model.named_steps["classifier"]
importances = clf.feature_importances_
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]

# Get encoded feature names
cat_features = ohe.get_feature_names_out(cat_cols)
all_features = list(cat_features) + num_cols

fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(fi_df["Feature"][:15], fi_df["Importance"][:15], color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")

# 8. Save model pipeline (includes encoder + model together)
joblib.dump({
    "model": model,
    "features": features
}, "model.joblib")

print("✅ Training complete. Model, report, and feature importance saved.")
