
# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

# Loading the dataset
data = pd.read_csv("C:/Users/sahil/Desktop/NLP_sentiment/Russia_ukrain_sentiment_analysis/filename.csv")
print(data.head())

# Printing the column names
print(data.columns)

# Selecting relevant columns from the dataset
data = data[["username", "tweet", "language"]]

# Checking for missing values
data.isnull().sum()

# Counting the occurrences of each language
data["language"].value_counts()

# Downloading stopwords for text preprocessing
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    # Convert text to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove punctuation marks
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove line breaks
    text = re.sub('\n', '', text)
    
    # Remove alphanumeric words
    text = re.sub('\w*\d\w*', '', text)
    
    # Remove stopwords
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    
    # Perform stemming on words
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    
    return text

# Apply the text cleaning function to the "tweet" column
data["tweet"] = data["tweet"].apply(clean)

# Generate a word cloud from the cleaned tweets
text = " ".join(i for i in data.tweet)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each tweet
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]

# Select the relevant columns for the final dataset
data = data[["tweet", "Positive", "Negative", "Neutral"]]

# Print the head of the dataset
print(data.head())

# Specify the path and filename for the new CSV file
output_file = "C:/Users/sahil/Desktop/NLP_sentiment/updated_data.csv"

# Save the updated dataset to the new CSV file
data.to_csv(output_file, index=False)

print("Updated dataset saved to", output_file)

