from preprocessing import *
from embeddings import *

############################## PREPROCESSING #################################

def preprocessing_pipeline(path, embedding_technique):

    print("Importing data...")
    df = import_data(path)

    print("Dropping na values..")
    df = na_values(df)

    print("Encoding classes..")
    df = class_encoding(df)

    print("Exploring length of articles..")
    article_len_exploration(df)

    print("Starting text preprocessing..")
    clean_text = preprocessing(df)

    df['clean_text'] = clean_text

    print(df.head())

    embedded_df = embedding(df, embedding_technique)

    return embedded_df

############################## EMBEDDING #################################

def embedding(df, embedding_technique):

    if embedding_technique == 'word2vec':

        print("Initialising Word2Vec vectorization")
        embedded_df = word2vec_vectorizer(df)
        embedded_df.to_csv('../models/embedding_data/word2vectest.csv')

    elif embedding_technique == 'tfidf':

        print("Initialising tf-idf vectorization")
        embedded_df = tf_idf_vectorizer(df)
        embedded_df.to_csv('../models/embedding_data/tfidftest.csv')

    elif embedding_technique == 'norm_bow':

        print("Initialising norm_bow vectorization")
        embedded_df = norm_bow_vectorizer(df)
        embedded_df.to_csv('../models/embedding_data/normbowtest.csv')

    else:
        print("ERROR. Please select one of the possible embedding techniques: word2vec, tfidf, normbow")


    return embedded_df

############################## MAIN #################################


path = '../labelled_data/labelled_data.csv'
embedding_technique = 'norm_bow'

preprocessing_pipeline(path, embedding_technique)

#TODO: change here the name of final output and check if path works
