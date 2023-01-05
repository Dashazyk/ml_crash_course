"""
----------------------------
Train model
----------------------------
"""

import os

import pickle

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from catboost import Pool, CatBoostClassifier

from src.utils import ML


if __name__ == '__main__':
    input_file = conf.raw_data_file
    if not os.path.exists(input_file):
        raise RuntimeError(f'No input file: {input_file}')
    df = pd.read_csv(input_file)
    train_df = df[df['subset'] == 'train']
    test_df = df[df['subset'] == 'test']
    logger.info('num rows for train: %d', train_df.shape[0])

    X_train = train_df['msg'].values
    y_train = train_df['label']

    X_test = test_df['msg'].values
    y_test = test_df['label']

    # preprocessing
    X_train = ML.preprocessing(X_train)
    X_test = ML.preprocessing(X_test)

    # vectorizing
    vectorizer = TfidfVectorizer(max_df=0.3, min_df=0.01).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    X_test_csr = vectorizer.transform(X_test)

    # fit 
    train_pool = Pool(
    X_train_csr, 
    y_train
    )
    valid_pool = Pool(
    X_test_csr, 
    y_test
    )

    catboost_params = {
        'iterations': 5000,
        'learning_rate': 0.01,
        'eval_metric': 'F1',
        'task_type': 'GPU',
        'early_stopping_rounds': 2000,
        'use_best_model': True,
        'verbose': 500
    }
    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool)

    # predict
    X_test_csr = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_csr)
    f1_score = f1_score(y_test, y_pred)

    logger.info('best_score %.5f', f1_score)

    # safe better model

    model_path = conf.model_path
    vectorizer_path = conf.vectorizer_path

    model.save_model(model_path)
    pickle.dump(vectorizer, open(vectorizer_path))

    logger.info("Saved model to disk")
