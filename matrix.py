import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("cyberbullying_tweets.csv", encoding="latin-1")

# ===== DATASET PREVIEW (ADDED) =====
print("DATASET PREVIEW:\n", df.head())
# ==================================

print(df.head())
print(df.info())

# Drop missing values
df = df.dropna()

# IMPORTANT: Change column names if needed
text_col = "tweet_text"
label_col = "cyberbullying_type"

# Features and labels
X = df[text_col]
y = df[label_col]

# Convert text to numbers using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

# ===== TF-IDF OUTPUT (ALREADY ADDED BEFORE) =====
print("TF-IDF Shape:", X.shape)
print("Sample Words:", tfidf.get_feature_names_out()[:10])
print("TF-IDF Sample Values:\n", X[0].toarray())
# ===============================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# ===== SAMPLE PREDICTIONS (ADDED) =====
print("\nSample Predictions:")
print(pd.DataFrame({
    "Actual": y_test.values[:5],
    "Predicted": y_pred[:5]
}))
# =====================================

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))