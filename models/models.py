import pandas as pd
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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


def evaluation(Y_test, Y_pred, model):
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
    print("\nAccuracy of " + model + " on test set : ", accuracy)

    print(cm)

    report = classification_report(Y_test, Y_pred)

    print(report)

    precision = precision_score(Y_test, Y_pred, labels=[0], average="weighted")
    recall = recall_score(Y_test, Y_pred, labels=[0], average="weighted")

    eval_format_string = "\n'Model': '{}', 'Embedding': '{}',"\
        + "\n'Accuracy': {:>0.{display_precision}f},"\
        + "\n'Precision (class 0)': {:>0.{display_precision}f},"\
        + "'Recall (class 0)': {:>0.{display_precision}f},"\
        + ""
    print(eval_format_string.format(model, "TBC", accuracy, precision, recall, display_precision=3))

    return

def Dummy(X_train, Y_train, X_test, Y_test):

    model = 'stratified dummy'

    classifier = DummyClassifier(strategy="stratified", random_state=42)

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)

    return

def NaiveBayes(X_train, Y_train, X_test, Y_test):

    model = 'naive bayes'

    classifier = GaussianNB()

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    evaluation(Y_test, Y_pred, model)

    return

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
    param_grid = {
            "n_estimators": [30, 50, 70],
             "max_depth": [3, 5, None],
             }
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=kfold)
    grid.fit(X_train, Y_train)

    print("\nTuned {0} parameters: {1}".format(model, grid.best_params_))

    classifier = grid.best_estimator_
    # classifier.fit(X_train, Y_train)

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
