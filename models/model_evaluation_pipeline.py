from models import *





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
        print("Error, potential models are: onevsrest, onevsone, randomforest and adaboost")

    return


def run():
    embeddings = ['word2vec','tfidf','normbow']
    models = ['onevsrest', 'onevsone', 'randomforest','adaboost']

    for embedding in embeddings:
        print("\n\n##########\n\n{0}".format(embedding.upper()))
        X_train, Y_train, X_test, Y_test = import_embedded_data(embedding)
        for model in models:
            print("\n\n----------\n\n{0} >> {1}".format(embedding.upper(), model.upper()))
            fit_predict(model, X_train, Y_train, X_test, Y_test)