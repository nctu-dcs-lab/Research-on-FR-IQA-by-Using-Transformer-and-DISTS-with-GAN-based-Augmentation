import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'TID2013 scores_record/1.1.7 pred_scores.pickle',
    'TID2013 scores_record/1.7.4 pred_scores.pickle',
]
pred_scores_weight = (0.61246728, 0.66153169)

with open('TID2013 scores_record/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

weight_avg_pred_scores = np.zeros(pred_scores[0].size)

for scores, weight in zip(pred_scores, pred_scores_weight):
    weight_avg_pred_scores += scores * (weight / sum(pred_scores_weight))

plcc, srcc, krcc = calculate_correlation_coefficient(
    gt_scores,
    weight_avg_pred_scores
)

print(f'PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
