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

## Net Sentiment of Each Stock
![net_sentiment](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/9c04aab1-1b69-49ab-bd52-5611e2e32c66)

This bar chart suggests that, on balance, more tweets about these companies are positive than negative.

Google, both under its ticker symbol GOOGL (Class A shares) and GOOG (Class C shares), has the highest net sentiment, which indicates that Google-related tweets have the most positive bias among the companies analyzed. This could suggest that the public opinion on Twitter towards Google is generally favorable, which could be due to various factors such as successful product launches, positive press coverage, or strong financial performance, among other possibilities.

Amazon (AMZN) has the lowest net sentiment, although still positive. While this means that more positive tweets are being made about Amazon than negative ones, the margin is less significant than with the other companies. This could potentially be an area of concern. It might be helpful to delve deeper into the dataset or use other sources of data to understand why the sentiment towards Amazon is comparatively lower.

Apple (AAPL), Microsoft (MSFT), and Tesla (TSLA) fall between Google and Amazon in terms of net sentiment. They have a less positive bias in their tweets compared to Google but more compared to Amazon. This suggests a generally positive public opinion, but there might be some concerns or areas of improvement that are influencing the sentiment to be less positive than for Google.

Overall, the sentiment analysis suggests that these tech companies are viewed positively on Twitter, with varying degrees of positivity. It's important to remember that this analysis only provides a snapshot based on the data available, and it might not capture all the nuances of public opinion. Furthermore, sentiment analysis is just one tool among many that should be used when evaluating companies for potential investment decisions. Other important factors include financial performance, market trends, competitive analysis, etc.

# Sentiment Distribution

## Apple (AAPL)
![AAPL_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/555e01b9-fe58-4caa-9125-bc14f1f15a89)
Apple (AAPL) has a high number of neutral tweets (1300), followed by positive tweets (1200), and comparatively fewer negative tweets (400). This suggests that Apple's online discussion is primarily neutral or positive, with a small portion of negative sentiment. The proportion of negative sentiment is relatively small compared to the total tweet volume.

## Amazon (AMZN)
![AMZN_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/37472f7d-5418-43db-a650-4ae2c4a95213)
Amazon (AMZN) has the highest proportion of neutral tweets (350), followed by positive (275), and negative tweets (125). Although the overall net sentiment is positive, the proportion of negative tweets to total tweets is higher for Amazon compared to Apple. This might explain why Amazon had a lower net sentiment in the first chart.

## Tesla (TSLA)
![TSLA_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/19b1d058-5a13-449c-adda-a13466dec927)
Tesla (TSLA) shows a similar trend as Amazon, with the majority of tweets being neutral (400), then positive (250), and fewer negative ones (75). Tesla, like Amazon, has a larger share of negative tweets relative to its total tweet volume compared to Apple, which might influence its net sentiment score.

## Microsoft (MSFT)
![MSFT_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/360d9fc9-3b3d-4c05-a6d4-3a17d133b370)
Microsoft (MSFT) also follows this trend with most tweets being neutral (140), then positive (140), and negative (50). Despite having fewer tweets in total compared to the others, the proportion of positive to negative tweets is higher, indicating a generally positive sentiment.

## Google (GOOG)
![GOOG_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/01f09d93-bbe6-431e-9eb7-36dcd43dde1f)
Based on the data provided, there were approximately 300 neutral, 500 positive, and 100 negative tweets related to Google under its ticker symbol GOOG.

This indicates that the majority of tweets relating to GOOG are positive, which is consistent with the high net sentiment observed earlier. The proportion of negative tweets is much lower, which further signifies a strongly positive sentiment among Twitter users towards GOOG.

The fact that the number of neutral tweets is lower than the positive tweets indicates that there are more defined (positive or negative) opinions than ambivalent or neutral sentiments about GOOG on Twitter.

## Google (GOOGL)
![GOOGL_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/3d1b6551-8dac-4502-8906-6a775d470669)
Similarly, for the ticker symbol GOOGL, there were approximately 150 neutral, 450 positive, and 25 negative tweets.

The pattern here is very similar to GOOG, with the majority of tweets being positive. However, it is important to note that the number of negative tweets for GOOGL is even less than for GOOG. This further solidifies the positive sentiment towards Google.

The proportion of neutral tweets is also lower than the positive tweets, indicating more defined opinions towards GOOGL, most of which are positive.

## Sentiment Distribution Summmary
In conclusion, Apple and Google have the strongest positive sentiment based on this analysis. Although Amazon and Tesla have positive net sentiments, the proportion of their negative tweets is higher, suggesting more mixed opinions about these companies. Microsoft has fewer tweets overall, but the sentiment is generally positive. These insights can be useful for understanding public opinion towards these companies and can potentially influence strategic decisions. However, this analysis should be complemented with other sources of data to provide a comprehensive understanding of the market sentiment.

## Correlation Matrix
![correlation_heatmap](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/e8655f37-e512-4592-a6c0-30f017c71db8)
A correlation matrix heatmap is generated to visualize the relationships between different sentiment metrics for each stock. This includes the count of positive, negative, and neutral tweets, the mean sentiment, net sentiment, and the total number of tweets.

## Word Cloud
![wordcloud](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/fe279cb7-195d-45b0-8cfe-783486365c8c)
Finally, a word cloud is generated to represent the most frequently occurring words in the processed tweets. This visualization helps identify common themes or topics in the tweet corpus.

## Summary 
This tool provides an excellent example of how to leverage NLP techniques, machine learning, and data visualization to derive insights from Twitter data in the financial market context. It could provide significant value in market research, investment decisions, or simply to track public opinion about different companies.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Side Note: 
The data sets are too large to upload to Github 
