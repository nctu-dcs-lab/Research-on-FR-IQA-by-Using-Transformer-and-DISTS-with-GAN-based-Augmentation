import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPRegressor

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

X = np.stack((np.concatenate((pred_scores[0]['train'], pred_scores[0]['val'])),
              np.concatenate((pred_scores[1]['train'], pred_scores[1]['val'])))).T
y = np.concatenate((gt_scores['train'], gt_scores['val']))
test_fold = np.concatenate((np.zeros(gt_scores['train'].size), -np.ones(gt_scores['val'].size)))

ps = PredefinedSplit(test_fold)
print(ps.get_n_splits())

X_val = np.stack((pred_scores[0]['val'], pred_scores[1]['val'])).T
y_val = gt_scores['val']

X_test = np.stack((pred_scores[0]['test'], pred_scores[1]['test'])).T
y_test = gt_scores['test']

parameters = {
    'solver': ('sgd', 'adam'),
    'alpha': (0.001, 0.0001, 0.00001),
    'learning_rate': ('constant', 'adaptive'),
    'learning_rate_init': (0.01, 0.001, 0.0001)}

mlp_regr = MLPRegressor(max_iter=500)

regr = GridSearchCV(mlp_regr, parameters, verbose=3, cv=ps)
regr.fit(X, y)

X_val_pred = regr.predict(X_val)

plcc, srcc, krcc = calculate_correlation_coefficient(
    y_val,
    X_val_pred
)

print(f'val PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')

X_test_pred = regr.predict(X_test)

plcc, srcc, krcc = calculate_correlation_coefficient(
    y_test,
    X_test_pred
)

print(f'test PLCC: {plcc: .4f}, SRCC: {srcc: .4f}, KRCC: {krcc: .4f}')
