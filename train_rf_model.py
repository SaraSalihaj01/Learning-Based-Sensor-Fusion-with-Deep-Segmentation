import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_rf():
    #Load the dataset generated with the 1.8m depth threshold
    csv_path = "fusion_dataset_seg.csv"
    print(f"[INFO] Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found. Please run create_fusion_dataset_seg.py first.")
        return

    #Feature selection
    # We use Edge Density (from U-Net), Distance (from Depth map), and Confidence (P_u)
    features = ['edge_density', 'd_u', 'P_u']
    X = df[features]
    y = df['label']

    #Dataset splitting for validation
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Model creation and training
    # e use 100 estimators to ensure stability in the STOP/GO decision
    print("[INFO] Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    #Model Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Model Accuracy: {acc:.4f}")
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred))

    #Save the trained model to a .joblib file
    model_name = "rf_fusion_model.joblib"
    joblib.dump(rf, model_name)
    print(f"[SUCCESS] Model saved as: {model_name}")

if __name__ == "__main__":
    train_rf()