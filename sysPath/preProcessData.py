import regex as re
import nltk
import warnings
import numpy as np
import pyarabic.normalize as Normalize

from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder

from matplotlib import style

le = LabelEncoder()
style.use("ggplot")



# nltk.download("punkt")  # download punkt tokenizer if not already downloaded
# nltk.download("wordnet")
# nltk.download("stopwords") # download stopwords if not already downloaded
# nltk.download("averaged_perceptron_tagger")
warnings.filterwarnings(action = 'ignore')



def remove_emojis(text):
    emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    u"\U0001F90C-\U0001F93A"  # Supplemental Symbols
                                    u"\U0001F93C-\U0001F945"  # and
                                    u"\U0001F947-\U0001F9FF"  # Pictographs
                                "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)



def removeConsecutiveDuplicates(text):
    # Replace any group of two or more consecutive characters with just one
    clean = re.sub(r'(\S)(\1{1,})', r'\1', text, flags=re.UNICODE)

    # Remove consecutive duplicate words
    clean = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', clean)

    return clean



def removeEnglish(text):
    return re.sub(r"[A-Za-z0-9]+"," ",text)



def lemmatizeArabic(text):
    """
    This function takes an Arabic word as input and returns its lemma using NLTK's ISRI stemmer
    """
    # Create an instance of the ISRI stemmer
    stemmer = ISRIStemmer()
    # Apply the stemmer to the word
    lemma = stemmer.stem(text)
    return lemma



def removeStopwords(text):
    # Tokenize the text into wordsz
    words = nltk.word_tokenize(text)
    # Get the Arabic stop words from NLTK
    stop_words = set(stopwords.words('arabic'))
    # Remove the stop words from the list of words
    words_filtered = [word for word in words if word.lower() not in stop_words]
    # Join the words back into a string
    clean = ' '.join(words_filtered)
    return clean



def removePunctuation(text):
    # Define the Arabic punctuation regex pattern
    arabicPunctPattern = r"[؀-؃؆-؊،؍؛؞]"
    engPunctPattern = r"[.,;''`~:\"]"

    # Use re.sub to replace all occurrences of Arabic punctuation with an empty string
    clean = re.sub(f"{arabicPunctPattern}|{engPunctPattern}", " ", text)
    return clean



def tokenizeArabic(text):
    # Tokenize the text using the word_tokenize method and return the list of tokenized words
    return word_tokenize(text)



def removeEmpty(dataset):
    indices_to_drop = []
    for index, tweet in dataset.iterrows():
        if not tweet['tweet'].strip():
            indices_to_drop.append(index)

    dataset = dataset.drop(index=indices_to_drop)
    dataset = dataset.reset_index(drop=True)
    return dataset



def cleanData(dataset):
    dataset = dataset.drop_duplicates(subset=["tweet"])
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    cleaned_tweets = []
    for tweet in dataset["tweet"]:
        #standard tweet cleaning

        # Remove URLs
        clean = re.sub(r"https?://\S+|www\.\S+", " ", tweet)

        # Remove punctuation, numbers, Arabic numbers, underscores
        clean = re.sub(r"\p{Mn}", "", clean) # Remove non-spacing marks
        clean = re.sub(r"\p{P}|[^\s\w\u0660-\u0669]|ﷻ|ﷺ|ۥۦ", " ", clean) # Replace punctuation, non-(whitespace/word) characters, and Arabic-Indic digits with spaces

        # Remove extra whitespaces
        clean = re.sub(r"\s+", " ", clean.strip())
        
        #Test to see if they're useful or not
        clean = remove_emojis(tweet)
        clean = removeConsecutiveDuplicates(clean)

        # mandatory arabic preprocessing
        clean = Normalize.normalize_searchtext(clean)
        clean = removeEnglish(clean)
        clean = lemmatizeArabic(clean)
        clean = removeStopwords(clean)
        clean = removePunctuation(clean)

        clean = re.sub(r"[^\u0621-\u064A\u0623\u0624\u0625\u0626\s]+", " ", clean)
        clean = re.sub(r"\s+", " ", clean.strip())

        cleaned_tweets.append(clean)

    # Replace the 'tweet' column with the cleaned tweets
    dataset["tweet"] = cleaned_tweets

    # Remove rows with empty or whitespace-only tweets
    dataset = removeEmpty(dataset.copy(deep = True))

    return dataset



def tokenization(dataset):
    for index, tweet in enumerate(dataset[["tweet"]].values.tolist()):
        if len(tweet) == 0 or len(tweet) == 1: continue

        tokenizedTweet = tokenizeArabic(*tweet)
        dataset.at[index, "tweet"] = tokenizedTweet

    dataset["dialect"] = le.fit_transform(dataset["dialect"])
    dataset["sarcasm"] = le.fit_transform(dataset["sarcasm"])

    return dataset



def preProcessData(dataset):    
    dataset = cleanData(dataset.copy(deep=True))
    print("\n\t----------        cleanData Done!        ----------\n")

    dataset = tokenization(dataset.copy(deep=True))
    print("\n\t---------     dataTokenization Done!     ----------\n")

    return dataset