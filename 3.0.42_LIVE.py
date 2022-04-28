import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'LIVE scores_record/1.1.7 pred_scores.pickle',
    'LIVE scores_record/1.3.2 pred_scores.pickle',
    'LIVE scores_record/1.5.1 pred_scores.pickle',
    'LIVE scores_record/1.7.4 pred_scores.pickle',
    'LIVE scores_record/2.2.3.3 pred_scores.pickle',
]
pred_scores_weight = (0.65749311, 0.03410907, 0.08268939, 0.48501394, 0.97157873)

with open('LIVE scores_record/gt_scores.pickle', 'rb') as handle:
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
