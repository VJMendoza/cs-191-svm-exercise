import pandas as pd
import numpy as np


def score_summary(grid, sort_by='mean_score'):
    def row(scores, params):
        d = {
            'estimator': key,
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
        }
        return pd.Series({**params, **d})

    rows = []
    params = grid.cv_results_['params']
    scores = []

    for i in range(grid.cv):
        key = "split{}_test_score".format(i)
        r = grid.cv_results_[key]
        scores.append(r.reshape(len(params), 1))

    all_scores = np.hstack(scores)
    for p, s in zip(params, all_scores):
        rows.append((row(s, p)))

    df = pd.concat(rows, axis=1, sort=True).T.sort_values(
        [sort_by], ascending=False)

    columns = ['estimator', 'min_score',
               'mean_score', 'max_score', 'std_score']
    columns = columns + [c for c in df.columns if c not in columns]

    return df[columns]
