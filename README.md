# Stock Sentiment Analysis using Twitter Data
This Python program analyzes sentiment towards various stocks using data from tweets. The sentiment analysis is done using the Natural Language Toolkit (NLTK). The result is a representation of sentiment towards the analyzed stocks, as well as visualizations in the form of a Word Cloud and sentiment distribution for each stock. This program uses NLTK to decypher 5000 tweets between the 6 companies is neutral, positive or negative.

## Dependencies
This project requires Python 3 and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-Learn
- WordCloud

## Data
The program uses two datasets: 'tweet.csv' and 'company_tweet.csv'. The 'tweet.csv' file contains the tweets and their corresponding tweet IDs, while the 'company_tweet.csv' file contains the ticker symbols and their corresponding tweet IDs.

## How to Run
Make sure you have installed all the dependencies mentioned above.
Place 'tweet.csv' and 'company_tweet.csv' in your working directory.
Run the Python script in your terminal/command prompt.
## Methodology
The program starts by loading and merging two datasets based on 'tweet_id'.

It then uses NLTK's SentimentIntensityAnalyzer to calculate sentiment scores for each tweet and labels them as 'positive', 'negative' or 'neutral'. This sentiment information is saved as a CSV file named 'stocks.csv'.

The tweets are then preprocessed: all non-word characters are removed, single characters are removed, multiple spaces are replaced by a single space, and all text is converted to lowercase.

The program then creates a Bag of Words model, using the processed tweets, and splits the data into a training set and a test set.

A Naive Bayes classifier is trained on this data and is then used to predict the sentiment of the test data.

Finally, the program calculates the net sentiment for each stock, and visualizes the sentiment distribution for each stock, the net sentiment for all stocks, and the correlation matrix heatmap. A Word Cloud is also generated using all the words in the tweets.

## Output
The output of the program includes:

A CSV file ('stocks.csv') containing tweets and their respective sentiments for different stocks.
A bar plot representing the net sentiment for each stock (green for positive sentiment and red for negative sentiment).
Bar plots for each stock, showing the sentiment distribution ('positive', 'negative', 'neutral').
A heatmap of the correlation matrix.
A Word Cloud representing frequently used words in the tweets.
Please note that due to the randomness of train-test split, the results may slightly vary each time you run the script.

# Analysis

## Net Sentiment 

![net_sentiment](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/1cd78739-fc14-4e85-bfd0-c666e4e55b00)

The bar chart shows that the stock with the most positive net sentiment is GOOGL, followed by GOOG and AAPL. The stock with the most negative net sentiment is AMZN, followed by FB and AMD.

The net sentiment for each stock is calculated by subtracting the number of negative tweets from the number of positive tweets, and then dividing by the total number of tweets. For example, the net sentiment for AAPL is calculated as follows:

(Number of positive tweets for AAPL) - (Number of negative tweets for AAPL) / (Total number of tweets for AAPL)

The net sentiment for a stock can be interpreted as the overall sentiment of the tweets about that stock. A positive net sentiment indicates that the tweets about the stock are mostly positive, while a negative net sentiment indicates that the tweets about the stock are mostly negative.

The bar chart shows that the overall sentiment of the tweets about the stocks in this dataset is positive. However, there are a no stocks with a negative net sentiment.

## Sentiment Distribution

## Apple
![AAPL_distribution](https://github.com/WillNaf/Twitter_sentiment/assets/118142412/ebce896f-fe40-46e7-99a7-665c6f4014a2)


## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Side Note: 
The data sets are too large to upload to Github 
