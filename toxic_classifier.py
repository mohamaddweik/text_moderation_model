import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 25k labeled data
df = pd.read_csv("labeled_data.csv")

# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# adding a column where toxic (1) and clean (0)
df['is_toxic'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)

# Text preprocessing function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only letters
    return text.lower()
    
df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)

# Define features (X) and labels (y)
X = df['clean_tweet']
y = df['is_toxic']
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train) #learns the vocabulary from training data
X_test_vectorized = vectorizer.transform(X_test) #converts test data using the same vocabulary

# Train the Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# print(df.head())
