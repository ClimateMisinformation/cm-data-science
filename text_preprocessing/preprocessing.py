import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import emoji
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


stop_words = set(stopwords.words('english'))
punctuation = string.punctuation.replace("-", "")
punctuation = punctuation.replace("!", "")
punctuation = punctuation.replace("?", "")
punctuation = punctuation.replace(".", "")


def import_data(filepath):
    '''
    Imports data as df from csv

    Parameters:
    filepath (str): path to csv to be imported

    Returns:
    df: data as dataframe object
    '''

    df = pd.read_csv(filepath, header=0)

    df.drop(df.columns.difference(['text', 'label']), 1, inplace=True)

    print(df.head())

    print('Size of dataframe')
    print(df.columns)
    print(df.shape)

    return df



# Counting NA values and transforming NA to empty strings
def na_values(df):

    '''
    Removed na values from column 'text'

    Parameters:
    df (dataframe): dataset as dataframe object

    Returns:
    df: dataframe object without na values
    '''

    print("NaN values per column")
    print(df.isna().sum())

    df.dropna(subset=['text'], inplace=True)
    df = df.replace(np.nan, '', regex=True)
    return df


def class_encoding(df):
    '''

    Drops all articles labelled with classes which are not of interest
    and encodes the following classes:

    - 118: climate denying -> encoded as 0
    - 119: not climate denying -> encoded as 1
    - 120: not climate related -> encoded as 2

    Saves encoded classes in column 'human_label'.
    Uses sklearn label encoder.

    Parameters:
    df (dataframe): dataset as dataframe object

    Returns:
    df: dataframe object with encoded classes.
    label_columns: list of column names holdng labels.
    '''

    # Multi classes.
    valid_classes = [118,119,120]
    df_encoded = df[df['label'].isin(valid_classes)]

    multi_class_values = df_encoded['label'].tolist()
    le.fit(multi_class_values)
    encoded_classes = le.transform(multi_class_values)
    unique_classes = le.classes_
    for x in unique_classes:
        print(str(x) + " is encoded to " + str(le.transform([x])))
    df_encoded['human_label'] = encoded_classes

    # Binary classes.
    df_encoded['human_binary_label'] = df_encoded['human_label'].apply(lambda label: 1 if label > 0 else 0)

    df_encoded = df_encoded.drop('label', axis=1)
    return df_encoded, ['human_label', 'human_binary_label']

def length_historgram(df):
    '''
    Explore the lengths of articles, stored in column 'text_lens'

    Prints description of 'text_lens' (average, max, min..).
    Plots histogram.

    Parameters:
    df (dataframe): dataset as dataframe object.
    '''

    print('Article length distributon')
    print(df['text_length'].describe())

    print('Histogram lens of articles')
    plt.hist(df['text_length'], bins=50)
    plt.xlabel('len articles')
    plt.show()

    return



# Exploring len of documents and filters documents over 1500 words.
def documents_len_exploration_and_filter(df, max_len=1500):
    '''

    Splits document text and stores length.
    Filters out long documents (>1500 words)
    Calls histogram_exploration_lengths function before and after filtering

    Parameters:
    df (dataframe): dataset as dataframe object.
    max_len (int): articles of len > max_len will be filtered out.

    Returns:
    df: filtered dataframe object where all documents are < 1500 words

    '''

    df['text_length'] = df['text'].apply(lambda text: len(text.split()))

    length_historgram(df)
    df = df[df['text_length'] <= max_len]
    length_historgram(df)

    return df


#TODO: clean preprocessing code
#Basic preprocessing
def preprocessing(df):
    '''
    Text preprocessing of article text includes:
    - Tokenization
    - Remove URLS
    - Remove newline characters
    - Remove special characters
    - Remove punctuation
    - Remove digits
    - Calls advanced_text_cleaning function for each article

    Parameters:
    df (dataframe): dataset as dataframe object.

    Returns:
    list_clean_text: list of preprocessed articles
    '''

    texts = df['text'].tolist()

    list_clean_text = []
    for text in texts:
        text = text.lower()
        split_text = re.split('-| |\n', text)

        # Remove URLS
        split_text = [re.sub(
            '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
            '', sent) for sent in split_text]

        # Remove new line characters
        split_text = [re.sub('\s+', ' ', sent) for sent in split_text]
        split_text = [re.sub('\n', ' ', sent) for sent in split_text]

        # Remove all special characters
        split_text = [re.sub("[^A-Za-z0-9]+", " ", sent) for sent in split_text]

        # Clean punctuation
        table = str.maketrans(' ', ' ', punctuation)
        split_text_no_punct = [w.translate(table) for w in split_text]

        # Remove numbers
        clean_text = [x for x in split_text_no_punct if not (x.isdigit())]


        clean_text = advanced_text_cleaning(clean_text)

        list_clean_text.append(clean_text)

    return list_clean_text

def hasEmoji(inputString):
    return any(char in emoji.UNICODE_EMOJI for char in inputString)


def advanced_text_cleaning(clean_article):
    '''
    Advanced text preporocessing of article text
    - Removing unwanted characters and words
    - Removing emojis
    - Lemmatising
    - Removing stop words

    Parameters:
    clean_article (list): preprocessed, tokenised article.

    Returns:
    list: tokenised article after further processing
        '''
    advanced_clean_article = []
    for w in clean_article:

        w = w.replace("???s", "")
        w = w.replace("share", "")

        if hasEmoji(w) == True:
            w = ""

        regex = re.compile('[^a-zA-Z]')
        w = regex.sub('', w)

        listWordsLemmatizer = ["us","has", "was", "as"]

        if w not in listWordsLemmatizer:
            w = lemmatizer.lemmatize(w)
            w = lemmatizer.lemmatize(w, pos='v')
            w = lemmatizer.lemmatize(w, pos='n')
            w = lemmatizer.lemmatize(w, pos='a')
        else:
            w = w

        w = w.replace("thisfacebooktwitterin", "")
        w = w.replace("facebooktwitterin", "")

        if w != '':
            advanced_clean_article.append(w)

    advance_cleaning_text = []

    #TODO: list comprehension below
    for w in advanced_clean_article:
        if w not in stop_words:
            advance_cleaning_text.append(w)

    #advance_cleaning_text = [w for w in advanced_clean_article if w not in stop_words]

    return advance_cleaning_text


