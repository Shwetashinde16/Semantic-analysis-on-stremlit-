import pandas as pd
import numpy as np
import re

import nltk

stop_words = {
    'i','me','my','myself','we','our','you','your','he','she','it','they',
    'is','am','are','was','were','be','been','being','have','has','had',
    'do','does','did','a','an','the','and','but','if','or','because','as',
    'until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from',
    'up','down','in','out','on','off','over','under'
}

from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report 

df=pd.read_csv("sentiment_dataset.csv")
print(df.head())
print(df.columns)

stemmer = PorterStemmer()
# stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

df['sentiment'] = df['sentiment'].map({
    'positive': 2,
    'neutral': 1,
    'negative': 0
})
print(df['sentiment'].isnull().sum())  # should print 0


X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



def predict_sentiment(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return "Positive" if prediction == 1 else "Negative"

print(predict_sentiment("I really love this product"))
print(predict_sentiment("Worst experience ever"))
