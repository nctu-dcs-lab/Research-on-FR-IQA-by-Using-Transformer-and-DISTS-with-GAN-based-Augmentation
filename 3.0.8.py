import pickle

import numpy as np

from src.tool.evaluate import calculate_correlation_coefficient

results = []
result_path = [
    'pred_scores/1.1.7 pred_scores.pickle',
    'pred_scores/1.3.2 pred_scores.pickle',
    'pred_scores/1.5.1 pred_scores.pickle',
    'pred_scores/2.2.3.2 pred_scores.pickle',
    'pred_scores/4.0.0 pred_scores.pickle'
]

for path in result_path:
    with open(path, 'rb') as handle:
        results.append(pickle.load(handle))

results = tuple(results)

gt_scores = {
    'val': results[0]['val']['gt_scores'],
    'test': results[0]['test']['gt_scores']
}

avg_pred = {
    'val': np.zeros(results[0]['val']['pred_scores'].size),
    'test': np.zeros(results[0]['test']['pred_scores'].size)
}

for mode in ['val', 'test']:
    for result in results:
        avg_pred[mode] += result[mode]['pred_scores']
    avg_pred[mode] /= len(results)

    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        avg_pred[mode]
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
