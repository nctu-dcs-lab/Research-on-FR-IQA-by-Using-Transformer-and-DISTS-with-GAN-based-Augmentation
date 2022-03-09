import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'scores_record/1.1.7 pred_scores.pickle',
    'scores_record/2.2.3.2 pred_scores.pickle',
]

with open('scores_record/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

weight_avg_pred_scores = {
    'train': np.zeros(pred_scores[0]['train'].size),
    'val': np.zeros(pred_scores[0]['val'].size),
    'test': np.zeros(pred_scores[0]['test'].size)
}

pred_scores_weight = (1/3, 2/3)

for mode in ['train', 'val', 'test']:
    for scores, weight in zip(pred_scores, pred_scores_weight):
        weight_avg_pred_scores[mode] += scores[mode] * weight

    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        weight_avg_pred_scores[mode]
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
