from models import *



def import_data(embedding):
    try:
        training_set = pd.read_csv("../labelled_data/embedded_data/" + embedding + "/train.csv", index_col=0)
        test_set = pd.read_csv("../labelled_data/embedded_data/" + embedding + "/test.csv", index_col=0)

        print("Training shape ", training_set.shape)
        print("Test shape ", test_set.shape)

        print("Training target distributions ")
        print(training_set['human_label'].value_counts())
        print("Test target distributions ")
        print(test_set['human_label'].value_counts())

    except Exception as e:
        print("Error loading embedding data. Possible embedding types are: word2vec, tfidf, bownorm")
        raise e

    X_train = training_set.iloc[:, :-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, :-1].values
    Y_test = test_set.iloc[:, -1].values

    return X_train, Y_train, X_test, Y_test

def fit_predict(model, X_train, Y_train, X_test, Y_test):

    if model == 'onevsrest':
        One_VS_Rest_SVM(X_train, Y_train, X_test, Y_test)
    elif model == 'onevsone':
        One_vs_One_SVM(X_train, Y_train, X_test, Y_test)
    elif model == 'randomforest':
        RandomForest(X_train, Y_train, X_test, Y_test)
    elif model == 'adaboost':
        AdaBoost(X_train, Y_train, X_test, Y_test)
    else:
        print("Error, potential models are: onevsrest, onevsone, randomfores and adaboost")

    return


embeddings = ['word2vec','tfidf','normbow']
models = ['onevsrest', 'onevsone', 'randomforest','adaboost']

for embedding in embeddings:
    print(embedding.upper())
    X_train, Y_train, X_test, Y_test = import_data(embedding)
    for model in models:
        fit_predict(model, X_train, Y_train, X_test, Y_test)