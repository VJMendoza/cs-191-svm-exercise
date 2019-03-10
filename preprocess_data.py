from bs4 import BeautifulSoup
from mailparser import parse_from_file
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from time import strftime
from datetime import datetime
from tqdm import tqdm
import logging
import multiprocessing
import numpy as np
import pandas as pd
import nltk
import os
import re
import string
import contractions

dataset_path = 'trec07p/'
index_file_path = 'full/index'
csv_path = 'processed-{}.csv'.format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))

saved_stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()


def read_data(index):
    index_file = pd.read_csv(index, sep=' ', names=['is_spam', 'email_path'])
    index_file['is_spam'] = index_file['is_spam'].map({'spam': 1, 'ham': 0})
    index_file['tokens'] = ''

    print('The file has been read.')
    return index_file


# This and the one below it came from Femo lol

def parallel_preprocess(df, num_processes=None):

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/
    with multiprocessing.Pool(num_processes) as pool:
        results = list(
            tqdm(pool.imap(parallel_preprocess_func, df.iterrows()),
                 total=df.shape[0],
                 unit='files',
                 dynamic_ncols=True))

    return results


def parallel_preprocess_func(d):
    row = d[1]

    try:
        email_path = os.path.join(
            dataset_path, index_file_path, '..', row['email_path'])
        email_path = os.path.abspath(email_path)
        email_body = preprocess_text(parse_from_file(email_path).body)
        if not email_body:
            row = None
        else:
            row['tokens'] = tokenize_text(email_body)
    except Exception:
        tqdm.write('Exception at {}'.format(row['email_path']))
        logging.exception('message')
        row = None

    return row


def preprocess_text(text):
    text = text.lower()

    text = BeautifulSoup(text, 'lxml').get_text().replace(
        '\n', ' ').replace('\r', '').strip()

    # https://github.com/kootenpv/contractions
    text = contractions.fix(text)

    text = punctuation_regex.sub('', text)

    return text


def tokenize_text(text):
    # Get list of tokens
    tokens = nltk.word_tokenize(text)

    # Remove words that are stopwords or numbers, lemmatize otherwise
    for index, token in enumerate(tokens):
        if token in saved_stopwords:
            tokens.pop(index)
        elif token.isdigit():
            tokens.pop(index)
        else:
            tokens[index] = lemmatizer.lemmatize(token)

    return tokens


def remove_empty_rows(rows):
    non_empty_rows = [x for x in rows if x is not None]

    num_empty = len(rows) - len(non_empty_rows)

    print('Removed {0} empty rows resulting to {1} from {2}'.format(
          num_empty, len(non_empty_rows), len(rows)))
    return non_empty_rows


if __name__ == '__main__':
    index_file = read_data(os.path.join(dataset_path, index_file_path))

    result_csv = parallel_preprocess(index_file)
    result_csv = remove_empty_rows(result_csv)
    result_csv = pd.DataFrame(result_csv)

    result_csv.to_csv(os.path.join(dataset_path, csv_path))
