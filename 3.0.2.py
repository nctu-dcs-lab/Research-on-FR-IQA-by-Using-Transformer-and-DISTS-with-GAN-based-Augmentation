import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'scores_record/1.1.7 pred_scores.pickle',
    'scores_record/1.3.2 pred_scores.pickle',
    'scores_record/1.5.1 pred_scores.pickle',
]

with open('scores_record/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

avg_pred_scores = {
    'val': np.zeros(pred_scores[0]['val'].size),
    'test': np.zeros(pred_scores[0]['test'].size)
}

for mode in ['val', 'test']:
    for scores in pred_scores:
        avg_pred_scores[mode] += scores[mode]
    avg_pred_scores[mode] /= len(pred_scores)

    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        avg_pred_scores[mode]
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
