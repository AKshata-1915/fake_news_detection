import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create dataset (small sample)
data = {
    "text": [
        "Government passes new education policy",
        "Shocking! Celebrity caught in secret scandal",
        "NASA announces new moon mission",
        "Breaking! Cure for cancer found overnight",
        "Elections results declared officially",
        "You won't believe what this politician said",
        "Scientists discover water on Mars",
        "Miracle weight loss in 2 days guaranteed"
    ],
    "label": [
        "REAL",
        "FAKE",
        "REAL",
        "FAKE",
        "REAL",
        "FAKE",
        "REAL",
        "FAKE"
    ]
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Step 5: Test accuracy
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Predict new news
def predict_news(news):
    news_vect = vectorizer.transform([news])
    result = model.predict(news_vect)
    return result[0]

# Try your own input
news_input = input("Enter news: ")
print("Prediction:", predict_news(news_input))