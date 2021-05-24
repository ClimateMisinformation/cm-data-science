from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import gensim
import gensim.downloader as api
import pickle


def dummy_fun(doc):
    return doc

'''
Abstract class for that wraps around vectorizers.
'''
class VectorizerClassBase():
    model = None
    default_pickle_path = None

    def print_debug_info(self):
        raise NotImplementedError

    def load(self, pickle_path=None):
        '''
        Loads model from pickle file.
        :param pickle_path: File to load from. If None, loads from default path.
        '''
        file_path = self.default_pickle_path if pickle_path is None else pickle_path
        self.model = pickle.load(open(file_path, 'rb'))

    def save(self, pickle_path=None):
        '''
        Saves model to pickle file
        :param pickle_path: File to save to. If None, saves to default path.
        '''
        file_path = self.default_pickle_path if pickle_path is None else pickle_path
        pickle.dump(self.model, open(file_path, 'wb+'))

    def fit(self, df, column_to_fit_on='clean_text'):
        '''
        Fits vectorizer on dataframe df.

        :param
          df: Pandas Dataframe containing examples.
          column_to_fit_on: name of column in df containing examples.
        '''
        raise NotImplementedError

    def run(self, df, column_to_run_on='clean_text',label_column=None):
        '''
        Runs vectorizer on dataframe df.

        :param df: Pandas Dataframe containing examples.
        :param column_to_run_on: name of column in df containing examples.
          label_column: name of column containing human labels to copy into output df. If None, does nothing.

        :return:
          dataframe containining embedded data.
        '''
        raise NotImplementedError


class Word2VecVectorizerClass(VectorizerClassBase):
    pickle_path = "./saved_vectorizers/Word2Vec_vectorizer.pkl"
    words_found = 0
    words_not_found = 0
    words_not_found_list = []

    # TODO(Renu): figure out if model can pickled
    def load(self):
        raise NotImplementedError

    # TODO(Renu): figure out if model can pickled
    def save(self):
        raise NotImplementedError

    def get_avg_word2vec(self,doc):
        '''
        Returns average of word2vec embeddings for document doc.
        :param doc: list of words in document
        :return: vector holding average of word2vec embeddings
        '''
        word_vectors = []
        for word in doc:
            try:
                vector = self.model.get_vector(word)
                word_vectors.append(vector)
                self.words_found += 1
            except KeyError:
                self.words_not_found += 1
                self.words_not_found_list.append(word)
        return np.mean(word_vectors, axis=0)

    def fit(self, df, column_to_fit_on='clean_text'):
        path = api.load('word2vec-google-news-300', return_path=True)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    def run(self, df, column_to_run_on='clean_text', label_column=None):
        # reinitialize counters
        self.words_found = 0
        self.words_not_found = 0
        self.words_not_found_list = []

        list_of_averages = df[column_to_run_on].apply(lambda doc: self.get_avg_word2vec(doc)).to_list()
        final_df = pd.DataFrame(list_of_averages)

        if label_column is not None:
            final_df[label_column] = df[label_column].to_list()
        return final_df

    def print_debug_info(self):
        print("words not found ", self.words_not_found)
        print("words found ", self.words_found)
        print("% of words not found ", (self.words_not_found / (self.words_not_found + self.words_found)) * 100)


class TfIdfVectorizerClass(VectorizerClassBase):
    pickle_path = "./saved_vectorizers/TfIdf_vectorizer.pkl"

    def fit(self, df, column_to_fit_on='clean_text'):
        self.model = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None, min_df=5)
        docs = df[column_to_fit_on].to_list()
        self.model.fit(docs)

    def run(self, df, column_to_run_on='clean_text', label_column=None):
        docs = df[column_to_run_on].to_list()
        sparse_vectors = self.model.transform(docs)
        flattened_vectors = [sparse_vector.toarray().flatten() for sparse_vector in sparse_vectors]

        final_df = pd.DataFrame(flattened_vectors)
        final_df.columns = self.model.get_feature_names()

        if label_column is not None:
            final_df[label_column] = df[label_column].to_list()
        return final_df

    def print_debug_info(self):
        print("Vocab length:", len(self.model.get_feature_names()))


class NormBowVectorizerClass(VectorizerClassBase):
    pickle_path = "./saved_vectorizers/NormBow_vectorizer.pkl"

    def fit(self, df, column_to_fit_on='clean_text'):
        self.model = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None, min_df=1,
            use_idf=False, norm='l2')
        docs = df[column_to_fit_on].to_list()
        self.model.fit(docs)

    def run(self, df, column_to_run_on='clean_text', label_column=None):
        docs = df[column_to_run_on].to_list()
        sparse_vectors = self.model.transform(docs)
        flattened_vectors = [sparse_vector.toarray().flatten() for sparse_vector in sparse_vectors]

        final_df = pd.DataFrame(flattened_vectors)
        final_df.columns = self.model.get_feature_names()

        if label_column is not None:
            final_df[label_column] = df[label_column].to_list()
        return final_df

    def print_debug_info(self):
        print("Vocab length:", len(self.model.get_feature_names()))