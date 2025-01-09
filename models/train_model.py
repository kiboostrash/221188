from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Dataset
texts = ["text1", "text2", "text3"]
labels = [0, 1, 0]

# Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Training
model = MultinomialNB()
model.fit(X, labels)

# Save model
with open("models/model.pkl", 'wb') as file:
    pickle.dump(model, file)
