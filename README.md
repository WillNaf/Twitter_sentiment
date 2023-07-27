# Stock Sentiment Analysis using Twitter Data
This project is a Python-based sentiment analysis tool for Twitter data. It focuses on tweets related to publicly traded stocks and uses various Natural Language Processing (NLP) and Machine Learning techniques to analyze the sentiment of these tweets.

## Table of Contents
- Requirements
- Data Preparation
- Sentiment Analysis
- Preprocessing
- Vectorization
- Model Training and Prediction
- Net Sentiment Calculation
- Data Visualization
- Correlation Matrix
- Word Cloud

## Requirements
The libraries used in this project are:

- re
- matplotlib
- numpy
- pandas
- nltk
- sklearn
- wordcloud
- seaborn
- os
Ensure these are installed in your environment before running the script.

## Data Preparation
The data required for this project include two CSV files: tweet.csv and company_tweet.csv. The first dataset consists of tweets, and the second dataset associates these tweets with particular publicly-traded companies based on ticker symbols. The datasets are loaded and merged on the 'tweet_id' field, associating each tweet with its respective company.

## Sentiment Analysis
Sentiment analysis is performed on each tweet using the SentimentIntensityAnalyzer from NLTK. This tool provides a polarity score, which is a float within the range of -1 (most negative) to +1 (most positive). These scores are then categorized as 'positive', 'negative', or 'neutral', based on their values.

## Preprocessing
Tweets undergo a preprocessing step to make them suitable for machine learning. These steps include removing non-word characters, single letters, multiple spaces, and converting all text to lowercase.

## Vectorization
Tweets are vectorized using the Bag-of-Words method. This process converts the processed text into numerical feature vectors, enabling their use in the machine learning model.

## Model Training and Prediction
The dataset is split into training (80%) and testing (20%) sets. The Multinomial Naive Bayes algorithm is then trained on the training set and used to predict the sentiment of the tweets in the test set.

## Net Sentiment Calculation
Net sentiment for each stock is calculated by subtracting the proportion of negative tweets from the proportion of positive tweets. This gives an overall sentiment value for each company.

## Data Visualization
The sentiment data is visualized in several ways, including:

A bar chart displaying the net sentiment for each stock.
A bar chart showing the distribution of sentiment (positive, negative, neutral) for each stock.
A correlation matrix heatmap illustrating the relationship between various sentiment metrics.
Correlation Matrix
A correlation matrix heatmap is generated to visualize the relationships between different sentiment metrics for each stock. This includes the count of positive, negative, and neutral tweets, the mean sentiment, net sentiment, and the total number of tweets.

## Word Cloud
Finally, a word cloud is generated to represent the most frequently occurring words in the processed tweets. This visualization helps identify common themes or topics in the tweet corpus.


## Summary 
This tool provides an excellent example of how to leverage NLP techniques, machine learning, and data visualization to derive insights from Twitter data in the financial market context. It could provide significant value in market research, investment decisions, or simply to track public opinion about different companies.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Side Note: 
The data sets are too large to upload to Github 
