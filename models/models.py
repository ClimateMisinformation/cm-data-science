import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


def split_train_test(data):


    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

    print("Training shape ", training_set.shape)
    print("Test shape ", test_set.shape)

    print("Training target distributions ")
    print(training_set['classes'].value_counts())
    print("Test target distributions ")
    print(test_set['classes'].value_counts())

    X_train = training_set.iloc[:, :-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, :-1].values
    Y_test = test_set.iloc[:, -1].values

    return X_train, Y_train, X_test, Y_test


def evaluation(Y_test, Y_pred, Y_prob, model, fancy_plots=True):
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
#     print("\nAccuracy of " + model + " on test set : ", accuracy)

    if fancy_plots:
        sns.heatmap(cm, square=False, annot=True, fmt='d', cbar=True, center=0)
        plt.xlabel('predicted')
        plt.ylabel('actual');
        plt.show()
    else:
        print(cm)

    report = classification_report(Y_test, Y_pred)
    print(report)

    precision = precision_score(Y_test, Y_pred, labels=[0], average="weighted")
    recall = recall_score(Y_test, Y_pred, labels=[0], average="weighted")
    if Y_prob is None:
        rocauc, logloss = None, None
    else:
        rocauc = roc_auc_score(Y_test, Y_prob, multi_class="ovr", average="macro")
        logloss = log_loss(Y_test, Y_prob)

    eval_dict = {"Model": model,
                "Accuracy": accuracy,
                "Precision (class 0)": precision,
                "Recall (class 0)": recall,
                "ROC AUC (ovr)": rocauc,
                "Cross-entropy loss": logloss,
                }

    return eval_dict

def Dummy(X_train, Y_train, X_test, Y_test):

    model = 'pick most frequent'

    classifier = DummyClassifier(strategy="most_frequent", random_state=42)

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    Y_prob = classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)


def NaiveBayes(X_train, Y_train, X_test, Y_test):

    model = 'naive bayes'

    classifier = GaussianNB()

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    Y_prob = classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)


def One_VS_Rest_SVM(X_train, Y_train, X_test, Y_test):
    model = 'one vs rest svm (tuned)'

    # classifier = OneVsRestClassifier(SVC(kernel='linear', class_weight='balanced', probability=True))
    param_grid = {
            "estimator__kernel": ["linear", "rbf"],
             "estimator__C": np.linspace(0.1, 100, num=1000),
             "estimator__gamma": np.linspace(0.0001, 10, num=10000),
             }
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    grid = RandomizedSearchCV(OneVsRestClassifier(SVC(random_state=42, class_weight='balanced', probability=True)), 
            param_grid, cv=kfold, scoring="roc_auc_ovr", n_iter=5)
    grid.fit(X_train, Y_train)

    print("\nTuned {0} parameters: {1}".format(model, grid.best_params_))

    classifier = grid.best_estimator_
    # classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    Y_prob = classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)


def One_vs_One_SVM(X_train, Y_train, X_test, Y_test):
    model = 'one vs one svm'

    classifier = OneVsOneClassifier(SVC(kernel='linear', class_weight='balanced', probability=True, decision_function_shape='ovo'))
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    Y_prob = None # classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)


def RandomForest(X_train, Y_train, X_test, Y_test):
    model = 'random forest (tuned)'

    # classifier = RandomForestClassifier(n_estimators=30)
    param_grid = {
            "max_depth": range(5, 20, 5),
             "criterion": ["gini", "entropy"],
             "max_features": ["auto", "sqrt", 0.2, 0.3, 0.4, 0.5],
             }
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    grid = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_estimators=100), 
            param_grid, cv=kfold, scoring="roc_auc_ovr", n_iter=100)
    grid.fit(X_train, Y_train)

    print("\nTuned {0} parameters: {1}".format(model, grid.best_params_))

    classifier = grid.best_estimator_
    # classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    Y_prob = classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)


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
    Y_prob = classifier.predict_proba(X_test)

    return evaluation(Y_test, Y_pred, Y_prob, model)

