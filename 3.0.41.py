import pickle
import time

import numpy as np
from scipy.optimize import differential_evolution

from src.tool.evaluate import calculate_correlation_coefficient


def loss_function(weights):
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores['val'],
        np.average(stack_pred_scores['val'], axis=0, weights=weights)
    )
    score = plcc + srcc + krcc
    return 3 - score


pred_scores = []
pred_scores_path = [
    'scores_record/1.1.7 pred_scores.pickle',
    'scores_record/1.3.2 pred_scores.pickle',
    'scores_record/1.5.1 pred_scores.pickle',
    'scores_record/1.7.4 pred_scores.pickle',
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

for mode in ['train', 'val', 'test']:
    plcc, srcc, krcc = calculate_correlation_coefficient(
        gt_scores[mode],
        np.average(stack_pred_scores[mode], axis=0, weights=best_weights)
    )

    print(f'{mode} PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
