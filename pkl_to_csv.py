import os
import pickle
from pathlib import Path

import pandas as pd

root_dir = Path('../data/PIPAL(processed)')
label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}
pred_scores = (
    'IQT-L_pred_scores',
    'IQT-M_pred_scores',
    'IQT-H_pred_scores',
    'IQT-Mixed_pred_scores',
    'DISTS-Tune_pred_scores'
)


def get_df(dataset_type):
    tmp_df = []
    for filename in (root_dir / label_dir[dataset_type]).glob('*.txt'):
        df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
        tmp_df.append(df)

    df = pd.concat(tmp_df, axis=0, ignore_index=True)
    df['dist_img'] = df['dist_img'].apply(lambda x: x[:-4])
    df = df.sort_values('dist_img')
    df.rename(columns={'score': 'gt_score'}, inplace=True)
    df['dataset_type'] = dataset_type
    df.reset_index(inplace=True, drop=True)

    return df


for pred_score in pred_scores:
    with open(os.path.join('scores_record/PIPAL', pred_score + '.pickle'), 'rb') as f:
        scores_record = pickle.load(f)

    dfs = []
    for dataset_type in label_dir.keys():
        tmp_df = get_df(dataset_type)
        tmp_df['pred_score'] = scores_record[dataset_type]
        if pred_score != 'DISTS-Tune_pred_scores':
            tmp_df['pred_score'] = 1 - tmp_df['pred_score']
        dfs.append(tmp_df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.sort_values('dist_img', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df[['dist_img', 'dataset_type', 'gt_score', 'pred_score']]

    df.to_csv(os.path.join('scores_record/PIPAL', pred_score + '.csv'), index=False)
