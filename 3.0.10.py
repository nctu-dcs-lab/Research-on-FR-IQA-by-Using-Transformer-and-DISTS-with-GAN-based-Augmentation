import pickle

import numpy as np

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

for mode in ['train', 'val', 'test']:

    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        np.average(stack_pred_scores[mode], axis=0)
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')


