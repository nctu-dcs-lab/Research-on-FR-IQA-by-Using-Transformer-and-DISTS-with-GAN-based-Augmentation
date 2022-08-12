import pickle
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from DISTS_pytorch import DISTS
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient

root_dir = Path('../data/PIPAL(processed)')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netD = DISTS().to(device)

def get_PIPAL_df(dataset_type):
    root_dir = Path('../data/PIPAL(processed)')

    label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}

    tmp_df = []
    for filename in (root_dir / label_dir[dataset_type]).glob('*.txt'):
        df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
        tmp_df.append(df)

    df = pd.concat(tmp_df, axis=0, ignore_index=True)

    df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
    df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')
    df = df[['dist_img', 'ref_img']].sort_values('dist_img')

    return df

def get_pred_scores(df, netD, img_size, device):
    def transform(ref_img, dist_img, img_size):
        ref_img = TF.resize(ref_img, img_size)
        dist_img = TF.resize(dist_img, img_size)

        ref_img = TF.to_tensor(ref_img)
        dist_img = TF.to_tensor(dist_img)

        return ref_img, dist_img

    pred_scores_list = []

    for index, data in tqdm(df.iterrows()):
        ref_img = Image.open(data['ref_img']).convert('RGB')
        dist_img = Image.open(data['dist_img']).convert('RGB')
        ref_img, dist_img = transform(ref_img, dist_img, img_size)

        ref_img = ref_img.to(device)
        dist_img = dist_img.to(device)

        # Format batch
        c, h, w = ref_img.size()

        with torch.no_grad():
            """
            Evaluate distorted images
            """
            pred_scores = netD(ref_img.view(-1, c, h, w), dist_img.view(-1, c, h, w))

            # Record original predict scores
            pred_scores_list.append(pred_scores.view(-1).cpu().detach())

    pred_scores_arr = torch.cat(pred_scores_list).numpy()

    return pred_scores_arr


records = {}
for dataset_type in ['train', 'val', 'test']:
    df = get_PIPAL_df(dataset_type)
    records[dataset_type] = get_pred_scores(df, netD, (256, 256), device)

with open("DISTS_pred_scores", 'wb') as handle:
        pickle.dump(records, handle)
