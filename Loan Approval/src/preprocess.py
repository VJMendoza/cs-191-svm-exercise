from time import strftime
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from imputer import DataFrameImputer
import logging
import multiprocessing
import numpy as np
import pandas as pd
import nltk
import os
import re
import string
import contractions

dataset_path = Path("../data")
csv_path = "processed.csv"
dataset_file = "train_loanPrediction.csv"


def read_data(index):
    print("--- Loading data ---")
    data = pd.read_csv(index, sep=',')

    print('File `{}` has been read'.format(dataset_file))
    return data


def cat_encode(dataset, cols):
    """Makes multiple columns into categorical and label encodes them
    """
    for col in cols:
        dataset[col] = dataset[col].astype('category')
        dataset[col] = dataset[col].cat.codes
    return dataset


def preprocess(dataset):
    print("--Preprocessing Dataset--")
    dataset = dataset.drop(['Loan_ID'], axis=1)
    dataset = dataset.replace({"Dependents": {"3+": "3"}})
    dataset = DataFrameImputer().fit_transform(dataset)
    dataset = cat_encode(dataset, ['Gender', 'Married', 'Education',
                                   'Self_Employed', 'Credit_History',
                                   'Property_Area', 'Loan_Status'])
    return dataset


if __name__ == '__main__':
    dataset = read_data(dataset_path / dataset_file)
    result_csv = preprocess(dataset)
    result_csv.to_csv(os.path.join(dataset_path, csv_path))
