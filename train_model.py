# train_model.py
import numpy as np
import joblib
from detector import FakeReviewDetector
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def main():
    detector = FakeReviewDetector()

    df = detector.load_data('amazon_reviews.csv')
    print(f"✅ Reviews loaded: {len(df)}")

    print("Extracting traditional features...")
    traditional_features = detector.extract_features(df)  # Trains the vectorizer

    print("Generating embeddings...")
    embeddings = detector.get_embeddings(df['review'].tolist())

    X = np.hstack([traditional_features.values, embeddings])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    X_train_scaled = detector.scaler.fit_transform(X_train)
    X_test_scaled = detector.scaler.transform(X_test)

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save AFTER vectorizer is trained
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(detector.scaler, 'scaler.pkl')
    joblib.dump(detector, 'feature_extractor.pkl')
    print("✅ Model, scaler, and feature extractor saved.")

if __name__ == "__main__":
    main()
