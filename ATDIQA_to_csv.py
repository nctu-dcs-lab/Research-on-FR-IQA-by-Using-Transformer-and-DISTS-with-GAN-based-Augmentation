import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.tool.evaluate import calculate_correlation_coefficient

root_dir = Path('../data/PIPAL(processed)')
label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}


def get_df(dataset_type):
    tmp_df = []
    for filename in (root_dir / label_dir[dataset_type]).glob('*.txt'):
        df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
        tmp_df.append(df)

    df = pd.concat(tmp_df, axis=0, ignore_index=True)
    df['dist_img'] = df['dist_img'].apply(lambda x: x[:-4])
    df = df.sort_values('dist_img')
    df.rename(columns={'score': 'gt_score'}, inplace=True)
    df['dataset_type'] = dataset_type
    df.reset_index(inplace=True, drop=True)

    return df


def loss_function(weights):
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores['val'],
        np.average(stack_pred_scores['val'], axis=0, weights=weights)
    )
    score = plcc + srcc + krcc
    return 3 - score


pred_scores = []
pred_scores_path = [
    'scores_record/PIPAL/IQT-L_pred_scores.pickle',
    'scores_record/PIPAL/IQT-M_pred_scores.pickle',
    'scores_record/PIPAL/IQT-H_pred_scores.pickle',
    'scores_record/PIPAL/IQT-Mixed_pred_scores.pickle',
    'scores_record/PIPAL/DISTS-Tune_pred_scores.pickle',
]

with open('scores_record/PIPAL/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

stack_pred_scores = {
    'train': np.stack([pred_scores[idx]['train'] for idx in range(len(pred_scores))]),
    'val': np.stack([pred_scores[idx]['val'] for idx in range(len(pred_scores))]),
    'test': np.stack([pred_scores[idx]['test'] for idx in range(len(pred_scores))])
}

bound_w = [(0.0, 1.0) for _ in range(len(pred_scores))]

begin_time = time.time()
result = differential_evolution(loss_function, bound_w, maxiter=1000, tol=1e-7, disp=True, workers=-1)

best_weights = result['x']
print(f'Optimized Weights: {best_weights}')

# evaluate chosen weights
plcc, srcc, krcc = calculate_correlation_coefficient(
    gt_scores['val'],
    np.average(stack_pred_scores['val'], axis=0, weights=best_weights)
)
score = plcc + srcc + krcc
print(f'Optimized Weights Score: {score:.3f}')

end_time = time.time()

print(f'Execution time: {int((end_time - begin_time) // 3600)}:{int((end_time - begin_time) // 60)}:{int((end_time - begin_time) % 60)}')

dfs = []
for mode in ['train', 'val', 'test']:
    df = get_df(mode)
    df['pred_score'] = 1 - np.average(stack_pred_scores[mode], axis=0, weights=best_weights)
    dfs.append(df)

df = pd.concat(dfs, axis=0, ignore_index=True)
df.sort_values('dist_img', inplace=True)
df.reset_index(inplace=True, drop=True)
df = df[['dist_img', 'dataset_type', 'gt_score', 'pred_score']]
df.to_csv(os.path.join('scores_record/PIPAL', 'ATDIQA_pred_scores.csv'), index=False)
