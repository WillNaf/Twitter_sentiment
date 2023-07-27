# Stock Sentiment Analysis using Twitter Data
This Python program analyzes sentiment towards various stocks using data from tweets. The sentiment analysis is done using the Natural Language Toolkit (NLTK). The result is a representation of sentiment towards the analyzed stocks, as well as visualizations in the form of a Word Cloud and sentiment distribution for each stock.

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

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Side Note: 
The data sets are too large to upload to Github 
