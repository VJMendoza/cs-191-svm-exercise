import pandas as pd
import numpy as np
from grid_helper import score_summary
from classifier_switcher import ClfSwitcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC


dataset_size = 10000
dataset_path = 'trec07p/processed-emails.csv'
score_file = 'scores-{}.csv'.format(dataset_size)

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

    df = df.head(dataset_size)

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

    grid = GridSearchCV(pipe, param_grid, cv=5,
                        return_train_score=True, n_jobs=-1)

    grid.fit(x_train, y_train)

    print('--- Best Parameter ---')
    print(grid.best_params_)

    scores = score_summary(grid)

    scores.to_csv(score_file)

    print(scores)

    print("Detailed classification report using the best estimator:")

    prediction = grid.predict(x_test)
    print('Accuracy: {:.05%}'.format(np.mean(prediction == y_test)))
    print(classification_report(y_test, prediction))
