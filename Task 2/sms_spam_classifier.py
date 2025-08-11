
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only necessary columns
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svm', LinearSVC())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, 'sms_spam_classifier.pkl')
print('Model saved as sms_spam_classifier.pkl')

# Example usage
example_msg = ["Congratulations! You’ve won a free ticket."]
prediction = pipeline.predict(example_msg)[0]
print('Example message prediction:', 'Spam' if prediction == 1 else 'Ham')
