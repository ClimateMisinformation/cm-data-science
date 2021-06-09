import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report


def import_embedded_data(embedding, label_columns=['human_label','human_binary_label'], label='human_label'):
    '''
    Import embedding training and test data with labels.

    :param embedding: string corresponding to type of embedding (one of: 'word2vec', 'tfidf', 'bownorm')
    :param label_columns: names of all columns in the csv corresponding to labels.
    :param label: name of column holding label to exract (one of the label_columns).
    :return: 4 dataframes corresponding to data and labels for train and test.
    '''
    try:
        training_set = pd.read_csv("../labelled_data/embedded_data/" + embedding + "/train.csv", index_col=0)
        test_set = pd.read_csv("../labelled_data/embedded_data/" + embedding + "/test.csv", index_col=0)

        print("Training shape ", training_set.shape)
        print("Test shape ", test_set.shape)

        print("Training target distributions ")
        print(training_set[label].value_counts())
        print("Test target distributions ")
        print(test_set[label].value_counts())

    except Exception as e:
        print("Error loading embedding data. Possible embedding types are: word2vec, tfidf, bownorm")
        raise e

    X_train = training_set.drop(label_columns, axis=1).values
    Y_train = training_set[label].values
    X_test = test_set.drop(label_columns, axis=1).values
    Y_test = test_set[label].values

    return X_train, Y_train, X_test, Y_test

def evaluation(Y_test, Y_pred, Y_prob, model, fancy_plots=True, pos_label=0):
    '''
    Evaluatets model predictions and/or scores on provided grounotruth.
    Handles binary and multi-class cases.

    :param Y_test: array holding groundtruth (elements correspond to a column index of Y_prob predicting the class)
    :param Y_pred: array holding predictions (elements have same format as Y_test)
    :param Y_prob: 1D or 2D array holding scores per class
    :param model: string holding model name, used for reporting
    :param fancy_plots: bool to plot fancy plots or not
    :param pos_label: label to use as positive label in either multi-class or binary evaluation.
    :return: dict holding eval metrics
    '''
    # If binary problem, grab first col of Y_prob only.
    is_binary_problem = False
    if len(Y_prob.shape) > 1 and Y_prob.shape[1] == 2:
        is_binary_problem = True
        Y_prob = Y_prob[:, 1:2]
    if len(Y_prob.shape) == 1 or Y_prob.shape[1] == 1:
        is_binary_problem = True
    if is_binary_problem:
        print("Eval identified for binary problem.")

    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
    print("\nAccuracy of " + model + " on test set : ", accuracy)

    if fancy_plots:
        sns.heatmap(cm, square=False, annot=True, fmt='d', cbar=True, center=0)
        plt.xlabel('predicted')
        plt.ylabel('actual')
        plt.show()
    else:
        print(cm)

    report = classification_report(Y_test, Y_pred)
    print(report)

    rocauc, logloss = None, None
    if is_binary_problem:
        precision = precision_score(Y_test, Y_pred, pos_label=pos_label)
        recall = recall_score(Y_test, Y_pred, pos_label=pos_label)
        if Y_prob is not None:
            logloss = log_loss(Y_test, Y_prob)
            if pos_label == 0:
                print('inverting scores and labels for rocauc because pos_label is set to 0...')
                Y_prob = 1.0-Y_prob
                Y_test = 1.0 - Y_test
            rocauc = roc_auc_score(Y_test, Y_prob)
    else:
        precision = precision_score(Y_test, Y_pred, labels=[pos_label], average="macro")
        recall = recall_score(Y_test, Y_pred, labels=[pos_label], average="macro")
        if Y_prob is not None:
            logloss = log_loss(Y_test, Y_prob)
            rocauc = roc_auc_score(Y_test, Y_prob, multi_class="ovr", average="macro")

    eval_dict = {"Model": model,
                "Accuracy": accuracy,
                "Precision (class 0)": precision,
                "Recall (class 0)": recall,
                "ROC AUC": rocauc,
                "Cross-entropy loss": logloss,
                }

    return eval_dict

def One_VS_Rest_SVM(X_train, Y_train, X_test, Y_test):

    model = 'one vs rest svm'

    classifier = OneVsRestClassifier(SVC(kernel='linear', class_weight='balanced', probability=True))

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)

    return

def One_vs_One_SVM(X_train, Y_train, X_test, Y_test):
    model = 'one vs one svm'

    classifier = OneVsOneClassifier(SVC(kernel='linear', class_weight='balanced', probability=True, decision_function_shape='ovo'))
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)
    return

def RandomForest(X_train, Y_train, X_test, Y_test):
    model = 'random forest'


    #TODO: FINE TUNE RANDOM FOREST
    classifier = RandomForestClassifier(n_estimators=30)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)

    return

def AdaBoost(X_train, Y_train, X_test, Y_test):
    model = 'ada boost'

    classifier = OneVsRestClassifier(AdaBoostClassifier(
        base_estimator=None,
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=123))
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)
    return
