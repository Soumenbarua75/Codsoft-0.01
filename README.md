# Codsoft-0.01
For intern work.
Creator = Soumen Barua


# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re

# 2. Load Dataset
df = pd.read_csv("movies.csv")  # Update path if needed

# 3. Basic Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['plot'] = df['plot'].astype(str).apply(clean_text)
df = df[['plot', 'genre']].dropna()

# Optional: if multiple genres -> pick first genre
df['genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip())

# Encode labels numerically (for SVC)
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['genre'])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['plot'],
    df['genre_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['genre_encoded']
)

# 5. Build ML Pipeline
# Choose your model: Logistic Regression, MultinomialNB, or LinearSVC
model_choice = "logistic"  # options: 'logistic', 'naive_bayes', 'svm'

if model_choice == "logistic":
    classifier = LogisticRegression(max_iter=1000, C=1.0)
elif model_choice == "naive_bayes":
    classifier = MultinomialNB()
elif model_choice == "svm":
    classifier = LinearSVC()
else:
    raise ValueError("Invalid model choice.")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', classifier)
])

# 6. Train Model
pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = pipeline.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(
    y_test, y_pred, target_names=label_encoder.classes_
))

# 8. Predict Example
def predict_genre(plot_summary):
    plot_summary = clean_text(plot_summary)
    prediction = pipeline.predict([plot_summary])
    genre = label_encoder.inverse_transform(prediction)
    return genre[0]

# Example usage
sample_plot = "A young wizard attends a school of magic while facing dark forces."
predicted_genre = predict_genre(sample_plot)
print(f"\n🎬 Predicted Genre: {predicted_genre}")
