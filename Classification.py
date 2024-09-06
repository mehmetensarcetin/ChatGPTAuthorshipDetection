# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import string
# import re
# import joblib

# # Preprocess the text data
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(f"[{string.punctuation}]", "", text)
#     return text

# # Load the data
# file_path = "Sınıflandırma.xlsx"
# df = pd.read_excel(file_path)

# # Combine the columns into a single DataFrame
# gpt_texts = df['GPT3.5'].dropna().tolist()
# human_texts = df['İnsan'].dropna().tolist()

# # Create labels
# texts = gpt_texts + human_texts
# labels = [1] * len(gpt_texts) + [0] * len(human_texts)

# # Preprocess the text data
# texts = [preprocess_text(text) for text in texts]

# # Define the pipeline
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('scaler', StandardScaler(with_mean=False)),
#     ('svm', SVC(kernel='sigmoid', C=10, gamma='scale'))
# ])

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Train the model
# pipeline.fit(X_train, y_train)

# # Save the model to a specific directory
# model_save_path = 'svm_sınıflandırma.joblib'
# joblib.dump(pipeline, model_save_path)

# print(f"Model saved successfully at {model_save_path}.")



#-----------------------------------------------------------------------------------




# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load the data
# file_path = "Sınıflandırma.xlsx"
# df = pd.read_excel(file_path)

# # Combine the columns into a single DataFrame
# gpt_texts = df['GPT3.5'].dropna().tolist()
# human_texts = df['İnsan'].dropna().tolist()

# # Create labels
# texts = gpt_texts + human_texts
# labels = [1] * len(gpt_texts) + [0] * len(human_texts)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Define feature extraction methods
# feature_union = FeatureUnion([
#     ('tfidf', TfidfVectorizer()),
#     ('count', CountVectorizer())
# ])

# # Define the pipeline with VotingClassifier
# pipeline = Pipeline([
#     ('features', feature_union),
#     ('scaler', StandardScaler(with_mean=False)),
#     ('voting', VotingClassifier(estimators=[
#         ('rf', RandomForestClassifier()),
#         ('gb', GradientBoostingClassifier()),
#         ('svm', SVC(probability=True))
#     ], voting='soft'))
# ])

# # Define the parameter grid for RandomizedSearchCV
# param_grid = {
#     'features__tfidf__max_df': [0.8, 0.9, 1.0],
#     'features__tfidf__ngram_range': [(1, 1), (1, 2)],
#     'features__count__max_df': [0.8, 0.9, 1.0],
#     'features__count__ngram_range': [(1, 1), (1, 2)],
#     'voting__rf__n_estimators': [100, 200, 300],
#     'voting__rf__max_depth': [None, 10, 20, 30],
#     'voting__gb__n_estimators': [100, 200, 300],
#     'voting__gb__learning_rate': [0.01, 0.1, 0.2],
#     'voting__svm__C': [0.1, 1, 10],
#     'voting__svm__gamma': ['scale', 'auto']
# }

# # Perform RandomizedSearchCV to find the best parameters
# random_search = RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, n_iter=50)
# random_search.fit(X_train, y_train)

# # Print best parameters
# print(f"Best parameters: {random_search.best_params_}")

# # Train the final model with the best parameters
# best_model = random_search.best_estimator_
# best_model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)

# # Ask user if they want to save the model
# save_model = input("Do you want to save the model? (yes/no): ")

# if save_model.lower() == 'yes':
#     # Save the model to a specific directory
#     model_save_path = 'ensemble_sınıflandırma.joblib'
#     joblib.dump(best_model, model_save_path)
#     print(f"Model saved successfully at {model_save_path}.")
# else:
#     print("Model not saved.")



import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data
file_path = "Sınıflandırma.xlsx"
df = pd.read_excel(file_path)

# Combine the columns into a single DataFrame
gpt_texts = df['GPT3.5'].dropna().tolist()
human_texts = df['İnsan'].dropna().tolist()

# Create labels
texts = gpt_texts + human_texts
labels = [1] * len(gpt_texts) + [0] * len(human_texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define feature extraction methods
feature_union = FeatureUnion([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_df=0.9)),
    ('count', CountVectorizer(ngram_range=(1, 1), max_df=1.0))
])

# Define the pipeline with VotingClassifier and best parameters
pipeline = Pipeline([
    ('features', feature_union),
    ('scaler', StandardScaler(with_mean=False)),
    ('voting', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=None)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)),
        ('svm', SVC(C=10, gamma='scale', probability=True))
    ], voting='soft'))
])

# Train the final model with the best parameters
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Ask user if they want to save the model
save_model = input("Do you want to save the model? (yes/no): ")

if save_model.lower() == 'yes':
    # Save the model to a specific directory
    model_save_path = 'ensemble_sınıflandırma.joblib'
    joblib.dump(pipeline, model_save_path)
    print(f"Model saved successfully at {model_save_path}.")
else:
    print("Model not saved.")
