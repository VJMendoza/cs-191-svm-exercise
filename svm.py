from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from classifier_switcher import ClfSwitcher
from grid_helper import score_summary
import numpy as np
import pandas as pd

dataset_path = 'trec07p/processed-emails.csv'
score_file = 'scores.csv'

param_grid = [
    {
        'clf__estimator': [SVC()],
        'clf__estimator__C': [1, 10, 100, 1000],
        'clf__estimator__kernel': ['rbf'],
        'clf__estimator__gamma': [1e-3, 1e-4, 'scale']
    },
    {
        'clf__estimator': [LinearSVC()],
        'clf__estimator__C': [1, 10, 100, 1000]
    },
    {
        'clf__estimator': [SVC()],
        'clf__estimator__C': [1, 10, 100, 1000],
        'clf__estimator__kernel': ['poly'],
        'clf__estimator__degree': [2, 3, 4],
        'clf__estimator__gamma': [1e-3, 1e-4, 'scale']
    }
]

if __name__ == "__main__":
    df = pd.read_csv(dataset_path, header=0,
                     usecols=['is_spam', 'tokens'])
    df.dropna(how='any', subset=['tokens'], inplace=True)

    df = df.head(500)

    print('--- Data Loaded ---')

    print('{0} in total. {1} as spam and {2} as ham'.format(
        df.shape[0], len(df[df['is_spam'] == 1]), len(df[df['is_spam'] == 0])))

    x_train, x_test, y_train, y_test = train_test_split(
        df['tokens'], df['is_spam'], test_size=0.2, random_state=191)

    pipe = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', ClfSwitcher())
    ])

    grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True)

    grid.fit(x_train, y_train)

    print('The best parameters are: ')
    print(grid.best_params_)

    scores = score_summary(grid)

    scores.to_csv(score_file)

    print(scores)
