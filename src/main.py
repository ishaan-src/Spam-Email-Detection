# from data_preprocessing import load_and_preprocess_data

# X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/spam.csv")

# print(X_train.iloc[0])
# print(y_train.value_counts())


# from data_preprocessing import load_and_preprocess_data
# from feature_extraction import apply_tfidf

# X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/spam.csv")
# X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(X_train, X_test)

# print(X_train_tfidf.shape)
# print(X_test_tfidf.shape)


# from data_preprocessing import load_and_preprocess_data
# from feature_extraction import apply_tfidf
# from train_models import train_all_models

# X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/spam.csv")
# X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(X_train, X_test)

# models = train_all_models(X_train_tfidf, y_train)

# print("Models trained:", models.keys())









# from data_preprocessing import load_and_preprocess_data
# from feature_extraction import apply_tfidf
# from train_models import train_all_models
# from evaluate_models import evaluate_models

# def main():
#     X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/spam.csv")

#     X_train_tfidf, X_test_tfidf, _ = apply_tfidf(X_train, X_test)

#     models = train_all_models(X_train_tfidf, y_train)

#     results_df = evaluate_models(models, X_test_tfidf, y_test)

#     print("\nModel Comparison:\n")
#     print(results_df.sort_values(by="F1-score", ascending=False))

# if __name__ == "__main__":
#     main()





# yha se app wala

from data_preprocessing import load_and_preprocess_data
from feature_extraction import apply_tfidf
from train_models import train_all_models, save_models

X_train, X_test, y_train, y_test = load_and_preprocess_data("data/spam.csv")
X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(X_train, X_test)

models = train_all_models(X_train_tfidf, y_train)
save_models(models, vectorizer)

