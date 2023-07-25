import re
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
import seaborn as sns

# Load the 'tweets' dataset
df_tweets = pd.read_csv('tweet.csv')

# Use only the first 500 rows
df_tweets = df_tweets.head(5000)

# Load the 'company_tweets' dataset
df_company = pd.read_csv('company_tweet.csv')

# Merge the two datasets on 'tweet_id'
df = pd.merge(df_tweets, df_company, on='tweet_id')

# Instantiate SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Collect tweets and sentiment
tweets_data = []
sentiment_count = defaultdict(lambda: defaultdict(int))
sentiment_scores = defaultdict(list)

for index, row in df.iterrows():
    tweet_text = row['body']
    ticker_symbol = row['ticker_symbol']
    pol_score = sia.polarity_scores(tweet_text)
    sentiment_scores[ticker_symbol].append(pol_score['compound'])
    if pol_score['compound'] >= 0.05:
        sentiment = 'positive'
    elif pol_score['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    tweets_data.append([ticker_symbol, tweet_text, sentiment])
    sentiment_count[ticker_symbol][sentiment] += 1

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

# Display net sentiment for each stock with colors based on sentiment
plt.figure(figsize=(10, 5))
colors = ['green' if value >= 0 else 'red' for value in net_sentiment.values()]
plt.bar(net_sentiment.keys(), net_sentiment.values(), color=colors)
plt.title('Net sentiment for each stock')
plt.show()

# Display sentiment distribution for each stock
for stock, sentiment_dict in sentiment_count.items():
    sentiments = list(sentiment_dict.keys())
    counts = list(sentiment_dict.values())
    color_mapping = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
    colors = [color_mapping[sentiment] for sentiment in sentiments]
    plt.figure(figsize=(10, 5))
    plt.bar(sentiments, counts, color=colors)
    plt.title(f'Sentiment distribution for {stock}')
    plt.show()

mean_sentiment = {stock: np.mean(scores) for stock, scores in sentiment_scores.items()}

corr_data = {}

for stock, sentiment_dict in sentiment_count.items():
    total_tweets = sum(sentiment_dict.values())
    corr_data[stock] = {
        'count_good': sentiment_dict['positive'],
        'count_bad': sentiment_dict['negative'],
        'count_neutral': sentiment_dict['neutral'],
        'mean': mean_sentiment[stock],
        'net_sentiment': net_sentiment[stock],
        'total_tweets': total_tweets
    }

# Create the correlation matrix
corr_df = pd.DataFrame(corr_data).T
corr_matrix = corr_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation matrix heatmap')
plt.show()

# Create a Word Cloud
all_words = ' '.join([text for text in processed_tweets])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
