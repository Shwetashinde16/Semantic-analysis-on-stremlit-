import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# 1. Load FULL dataset (40 rows)
df = pd.read_csv("raw_text_data_no_emoji.csv")

print("Total rows in file:", len(df))  # should show 40


# 2. Training data (small example to teach model)
train_texts = [
    "I love this movie",
    "This product is amazing",
    "Very happy with the service",
    "I hate this product",
    "Worst experience ever",
    "This is terrible",
    "Very good and nice", 
    "Extremely bad quality" ]
 
train_labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "positive",
    "negative" ]

# 3. Build Model
model = make_pipeline(CountVectorizer(), MultinomialNB())


# 4. Train Model
model.fit(train_texts, train_labels)


# 5. Predict for ALL rows in CSV
df["predicted_sentiment"] = model.predict(df["text"])


# 6. Save and Show Results
df.to_csv("sentiment_predictions.csv", index=False)

print("\nPredictions for ALL rows:\n")
print(df[["id", "text", "predicted_sentiment"]])
