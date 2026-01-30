from data_preprocessing import load_and_preprocess_data
from feature_extraction import apply_tfidf
from train_models import train_all_models, save_models

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/spam.csv")

# TF-IDF
X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(X_train, X_test)

# Train models
models = train_all_models(X_train_tfidf, y_train)

# Save models
save_models(models, vectorizer)

print("✅ Models and vectorizer saved successfully")
