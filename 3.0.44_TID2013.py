import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'TID2013 scores_record/1.1.7 pred_scores.pickle',
    'TID2013 scores_record/1.3.2 pred_scores.pickle',
    'TID2013 scores_record/1.5.1 pred_scores.pickle',
    'TID2013 scores_record/1.7.4 pred_scores.pickle',
    'TID2013 scores_record/4.0.0 pred_scores.pickle',
]

with open('TID2013 scores_record/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

stack_pred_scores = np.stack([pred_scores[idx] for idx in range(len(pred_scores))])

plcc, srcc, krcc = calculate_correlation_coefficient(
    gt_scores,
    np.average(stack_pred_scores, axis=0)
)

print(f'PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
