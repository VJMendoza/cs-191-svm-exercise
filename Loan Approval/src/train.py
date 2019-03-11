from sklearn.svm import SVC
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from time import gmtime, strftime, time
import pandas as pd
import numpy as np

dataset_path = Path("../data")
csv_file = "processed.csv"
results_file = "results.csv"

param_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['rbf'], 'gamma': [
                  1e-3, 1e-4, 'auto'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'],  'gamma': [
                  1e-3, 1e-4, 'auto'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['sigmoid'],  'gamma': [
                  1e-3, 1e-4, 'auto'], 'C': [1, 10, 100, 1000]}
              ]


def read_data(index):
    print("--- Loading data ---")
    data = pd.read_csv(index, sep=',')

    print('File `{}` has been read'.format(csv_file))
    return data


def cross_validate(df):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    time_start = time()
    # df = df.head(25)
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, 1:10], df["Loan_Status"], test_size=0.5, random_state=0)
    print("# Tuning hyper-parameters")
    print()

    clf = GridSearchCV(SVC(), param_grid, cv=5)
    clf.fit(X_train, y_train)
    time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
    print('--- GridSearchCV took {} ---'.format(time_taken))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    results_df = pd.DataFrame(clf.cv_results_)
    results_df.to_csv(dataset_path / results_file)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


if __name__ == "__main__":
    df = read_data(dataset_path / csv_file)
    print("--- Performing Grid Search Cross Validation ---")
    cross_validate(df)
