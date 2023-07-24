import re
import tweepy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from wordcloud import WordCloud

# Twitter API credentials
consumer_key = CONSUMER_KEY
consumer_secret = CONSUMER_SECRET_KEY
access_token = ACCESS_KEY
access_token_secret = ACCESS_SECRET_KEY

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

queries = ['AAPL', 'TSLA', 'GOOG', 'AMZN']
max_tweets = 250

# Instantiate SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Collect real-time tweets and sentiment
tweets_data = []
sentiment_count = defaultdict(lambda: defaultdict(int)) 

for q in queries:
    for tweet in tweepy.Cursor(api.search_tweets, q=q, lang="en").items(max_tweets):
        tweet_text = tweet.text.encode('utf-8').decode()
        pol_score = sia.polarity_scores(tweet_text)
        if pol_score['compound'] >= 0.05:
            sentiment = 'positive'
        elif pol_score['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        tweets_data.append([q, tweet_text, sentiment])
        sentiment_count[q][sentiment] += 1

# Save tweets data to a CSV file
df = pd.DataFrame(tweets_data, columns=['Stock', 'Tweet', 'Sentiment'])
df.to_csv('stocks.csv', index=False)

print("Saved tweets data to 'stocks.csv'")

# Preprocessing
stop_words = set(stopwords.words('english'))
processed_tweets = []
for data in tweets_data:
    tweet = data[1]
    tweet = re.sub(r'\W', ' ', str(tweet))
    tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
    tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
    tweet = re.sub(r'^b\s+', '', tweet)
    tweet = tweet.lower()
    processed_tweets.append(tweet)

# Bag of Words
vectorizer = CountVectorizer(max_features=2500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(processed_tweets).toarray()

# Sentiments
sentiments = [data[2] for data in tweets_data]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=0)

# Training Naive Bayes Classifier
classifier = MultinomialNB().fit(X_train, y_train)

# Predicting sentiment of live tweets
live_predictions = classifier.predict(X_test)

# Calculating net sentiment for each stock
net_sentiment = {}
for stock, sentiment_dict in sentiment_count.items():
    total_tweets = sum(sentiment_dict.values())
    net_sentiment[stock] = ((sentiment_dict['positive'] / total_tweets) - (sentiment_dict['negative'] / total_tweets))

# Display net sentiment for each stock
plt.figure(figsize=(10, 5))
plt.bar(net_sentiment.keys(), net_sentiment.values(), color=['red', 'green', 'blue', 'purple'])
plt.title('Net sentiment for each stock')
plt.show()

# Display sentiment distribution for each stock
for stock, sentiment_dict in sentiment_count.items():
    sentiments = list(sentiment_dict.keys())
    counts = list(sentiment_dict.values())
    plt.figure(figsize=(10, 5))
    plt.bar(sentiments, counts, color=['red', 'green', 'blue'])
    plt.title(f'Sentiment distribution for {stock}')
    plt.show()

# Create a Word Cloud
all_words = ' '.join([text for text in processed_tweets])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
