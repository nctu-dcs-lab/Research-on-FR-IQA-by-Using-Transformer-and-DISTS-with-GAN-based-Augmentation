import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

pred_scores = []
pred_scores_path = [
    'LIVE scores_record/1.1.7 pred_scores.pickle',
    'LIVE scores_record/1.3.2 pred_scores.pickle',
    'LIVE scores_record/1.5.1 pred_scores.pickle',
    'LIVE scores_record/1.7.4 pred_scores.pickle',
    'LIVE scores_record/4.0.0 pred_scores.pickle',
]
pred_scores_weight = (0.29443265, 0.02250836, 0.02398765, 0.24275188, 0.33650384, 0.05174801, 0.02806762)

with open('LIVE scores_record/gt_scores.pickle', 'rb') as handle:
    gt_scores = pickle.load(handle)

for path in pred_scores_path:
    with open(path, 'rb') as handle:
        pred_scores.append(pickle.load(handle))

pred_scores = tuple(pred_scores)

weight_avg_pred_scores = np.zeros(pred_scores[0].size)

for scores, weight in zip(pred_scores, pred_scores_weight):
    weight_avg_pred_scores += scores * (weight / sum(pred_scores_weight))

num_type_map = {
    'jp2k': 227,
    'jpeg': 233,
    'wn': 174,
    'gblur': 174,
    'fastfading': 174
}

i = 0
for dist_type, num_dist in num_type_map.items():
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[i:i + num_dist],
        weight_avg_pred_scores[i:i + num_dist]
    )

    print(f'{dist_type}')
    print(f'PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
    print('-' * 10)
    i = i + num_dist
