import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------- Load Dataset ----------------
df = pd.read_csv("hand_sign_dataset.csv")

# Features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels A-Z -> 0-25
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------- XGBoost Data ----------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ---------------- Parameters ----------------
params = {
    "objective": "multi:softprob",   # softmax probabilities
    "num_class": len(label_encoder.classes_),  # 26 for A-Z
    "eval_metric": "mlogloss",       # logloss
    "eta": 0.05,                     # learning rate
    "max_depth": 6,                  # tree depth
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# ---------------- Train Model ----------------
evallist = [(dtrain, "train"), (dtest, "eval")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,            # max trees
    evals=evallist,
    early_stopping_rounds=20,        # stop if no improvement
    verbose_eval=50
)

# ---------------- Evaluation ----------------
y_pred_prob = bst.predict(dtest)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=label_encoder.classes_
))

# ---------------- Save Model ----------------
bst.save_model("hand_sign_xgb_model.json")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model + Label Encoder saved!")
