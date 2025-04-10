import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

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
    return text.lower().strip()
    
df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)

# Define features (X) and labels (y)
X = df['clean_tweet']
y = df['is_toxic']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train) #learns the vocabulary from training data
X_test_vectorized = vectorizer.transform(X_test) #converts test data using the same vocabulary

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3, n_jobs=-1, verbose=2,
    scoring='accuracy')

grid_search.fit(X_train_vectorized, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_vectorized)

print("Best parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()
# print(df.head())
