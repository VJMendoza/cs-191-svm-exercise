from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd

dataset_path = 'trec07p/processed-emails.csv'

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel':['rbf']}
]

scores = ['precision', 'recall']


def train_models(x_train, y_train):
    models = {}
    for score in scores:
        print('Tuning hyper-parameters for {}'.format(score))
        # clf = GridSearchCV(SVC(), param_grid, cv=3,
        #                    scoring='{}_macro'.format(score))

        clf = Pipeline([
            ('countvec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', GridSearchCV(SVC(), param_grid,
                                 cv=3, scoring='{}_macro'.format(score)))
        ])

        clf.fit(x_train, y_train)
        models[score] = clf

    return models
    # svm_model = Pipeline([
    #     ('countvect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', SVC(C=1, kernel='linear', gamma=1))
    # ])

    # svm_model.fit(x_train, y_train)

    # return svm_model


def eval_models(models, x_test, y_test):
    for score in scores:
        print('Best parameter set found:')
        print(models[score].best_params_)
        print('Grid scores:')
        means = models[score].cv_results_['mean_test_score']
        stds = models[score].cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, models[score].cv_results_['params']):
            print('{:.3f} (+/-{:.4f}) for {}', format(mean, std*2, params))

        print('Detailed classification report:')
        prediction = models[score].predict(x_test)
        print(classification_report(y_test, prediction))

    # print(model_name)
    # prediction = model.predict(x_test)
    # print('Accuracy: {:.05%}'.format(np.mean(prediction == y_test)))
    # print(classification_report(y_test, prediction))


if __name__ == "__main__":
    df = pd.read_csv(dataset_path, header=0,
                     usecols=['is_spam', 'tokens'])
    df.dropna(how='any', subset=['tokens'], inplace=True)
    print('--- Data Loaded ---')

    x_train, x_test, y_train, y_test = train_test_split(
        df['tokens'], df['is_spam'], test_size=0.2, random_state=191)

    print('--- Training Starting ---')
    email_classifier = train_models(x_train, y_train)
    print('--- Training Finished ---')

    eval_models(email_classifier, x_test, y_test)
