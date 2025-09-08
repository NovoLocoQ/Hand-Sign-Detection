import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # for saving the model

# Load shuffled dataset
df = pd.read_csv("hand_sign_dataset.csv")

# Split features (X) and labels (y)
X = df.drop("label", axis=1)   # all landmark coordinates
y = df["label"]                # alphabet labels

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest
clf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # let trees expand fully
    random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model for later use
joblib.dump(clf, "hand_sign_rf_model.pkl")
print("Model saved as hand_sign_rf_model.pkl")
