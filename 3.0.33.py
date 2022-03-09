import pickle
from itertools import product

import numpy as np
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'scores_record/1.1.7 pred_scores.pickle',
    'scores_record/1.3.2 pred_scores.pickle',
    'scores_record/1.5.1 pred_scores.pickle',
    'scores_record/1.7.4 pred_scores.pickle',
    'scores_record/2.2.3.2 pred_scores.pickle',
    'scores_record/4.0.0 pred_scores.pickle',
    'scores_record/5.1.0 pred_scores.pickle',
]

with open('scores_record/gt_scores.pickle', 'rb') as handle:
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

w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_score, best_weights = 0.0, None

for weights in tqdm(product(w, repeat=len(pred_scores))):
    if len(set(weights)) == 1:
        continue
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores['val'],
        np.average(stack_pred_scores['val'], axis=0, weights=weights)
    )
    score = plcc + srcc + krcc
    if score > best_score:
        best_score, best_weights = score, weights
        print(f'Best Score: {best_score}, Best Weights: {best_weights}')

for mode in ['train', 'val', 'test']:
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        np.average(stack_pred_scores[mode], axis=0, weights=best_weights)
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
