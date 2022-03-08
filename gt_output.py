import pickle
from pathlib import Path

import pandas as pd


def main():
    root_dir = Path('../data/PIPAL(processed)')
    label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}

    records = {}

    for dataset_type in ['train', 'val', 'test']:
        dfs = []
        for filename in (root_dir / label_dir[dataset_type]).glob('*.txt'):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
        df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')
        df = df[['dist_img', 'ref_img', 'score']].sort_values('dist_img')

        records[dataset_type] = df['score']

    with open('gt_output.pickle', 'wb') as handle:
        pickle.dump(records, handle)


if __name__ == '__main__':
    main()
