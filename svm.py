from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def learn(x_train, y_train):
    svm_model = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(C=1, kernel='linear', gamma=0))
    ])

    svm_model.fit(x_train, y_train)


if __name__ == "__main__":
    print('test')
