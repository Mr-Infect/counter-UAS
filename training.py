import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
DATA_DIR = "data"
MODEL_PATH = "model/ransomware_model.pkl"
SCALER_PATH = "model/feature_scaler.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

print("[INFO] Loading all CSV files from data directory...")
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

all_dataframes = []
labeled_dataframes = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df.drop(columns=['timestamp'], errors='ignore', inplace=True)

        known_cats = ['Protcol', 'Flag', 'Family', 'SeddAddress', 'ExpAddress', 'IPaddress', 'Threats']
        for col in known_cats:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        df.dropna(how='all', inplace=True)
        all_dataframes.append(df)

        if 'Prediction' in df.columns:
            labeled_dataframes.append(df)

        print(f"[✓] Processed file: {os.path.basename(file)}  Shape: {df.shape}")
    except Exception as e:
        print(f"[⚠️] Skipping {file}: {e}")

if not all_dataframes:
    raise ValueError("❌ No usable data found.")

combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"[INFO] Total Combined Dataset Shape: {combined_df.shape}")

# Only train on labeled data
if not labeled_dataframes:
    raise ValueError("❌ No labeled data found.")

labeled_df = pd.concat(labeled_dataframes, ignore_index=True)

# Align columns
print("[INFO] Aligning features across all datasets...")
all_columns = combined_df.drop(columns=["Prediction"], errors="ignore").columns
X = labeled_df.reindex(columns=all_columns, fill_value=0)
y = labeled_df["Prediction"]

# Final object-to-int encoding
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X.fillna(0, inplace=True)

# Fit scaler on aligned X
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("[INFO] Training model...")
model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=1,
    reg_lambda=2,
    #use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Save
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("[✅] Done. Model trained successfully.")
