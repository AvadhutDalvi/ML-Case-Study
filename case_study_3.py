# Email Spam Detection using Machine Learning

# Import required libraries
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

# Download stopwords (only required once)
nltk.download('stopwords')

# Load Dataset
# Dataset source: https://www.kaggle.com/uciml/sms-spam-collection-dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Display first few rows
print("Dataset Preview:\n", data.head())

# Data Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
data['cleaned_message'] = data['message'].apply(clean_text)

# Encode labels ('ham' = 0, 'spam' = 1)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_message'], data['label_num'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Multinomial Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with a new email
sample_email = ["Congratulations! You won a $1000 gift card. Click here to claim now."]
sample_tfidf = vectorizer.transform(sample_email)
prediction = model.predict(sample_tfidf)

print("\nPrediction for Sample Email:", "Spam" if prediction[0] == 1 else "Ham")
