import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class PIPAL(Dataset):
    def __init__(self, root_dir, mode='train', img_size=(192, 192)):
        dist_type = {
            '00': 0,
            '01': 12,
            '02': 12 + 16,
            '03': 12 + 16 + 10,
            '04': 12 + 16 + 10 + 24,
            '05': 12 + 16 + 10 + 24 + 13,
            '06': 12 + 16 + 10 + 24 + 13 + 14
        }

        label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}

        dfs = []
        for filename in (root_dir / label_dir[mode]).glob('*.txt'):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df['category'] = df['dist_img'].apply(lambda x: dist_type[x[6:8]] + int(x[9:11]))
        df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
        df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')

        self.origin_scores = df['score'].to_numpy()
        self.scores = 1 - ((self.origin_scores - np.min(self.origin_scores)) /
                           (np.max(self.origin_scores) - np.min(self.origin_scores)))

        self.categories = df['category'].to_numpy()

        self.df = df[['dist_img', 'ref_img']]

        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_img = Image.open(self.df['ref_img'].iloc[idx]).convert('RGB')
        dist_img = Image.open(self.df['dist_img'].iloc[idx]).convert('RGB')

        ref_img, dist_img = self.transform(ref_img, dist_img)

        return ref_img, dist_img, self.scores[idx], self.categories[idx], self.origin_scores[idx]

    def transform(self, ref_img, dist_img):
        if self.mode == 'train':
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(ref_img, output_size=self.img_size)
            ref_img = TF.crop(ref_img, i, j, h, w)
            dist_img = TF.crop(dist_img, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                ref_img = TF.hflip(ref_img)
                dist_img = TF.hflip(dist_img)

            rotate_angle = random.choice([0, 90, 180, 270])
            ref_img = TF.rotate(ref_img, rotate_angle)
            dist_img = TF.rotate(dist_img, rotate_angle)

            ref_img = TF.to_tensor(ref_img)
            dist_img = TF.to_tensor(dist_img)

            ref_img = TF.normalize(ref_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            dist_img = TF.normalize(dist_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            return ref_img, dist_img

        else:
            ref_imgs = TF.five_crop(ref_img, self.img_size)
            dist_imgs = TF.five_crop(dist_img, self.img_size)

            ref_imgs = torch.stack([TF.normalize(TF.to_tensor(crop),
                                                 [0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                                    for crop in ref_imgs])
            dist_imgs = torch.stack([TF.normalize(TF.to_tensor(crop),
                                                  [0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
                                     for crop in dist_imgs])

            return ref_imgs, dist_imgs


def create_dataloaders(cfg):
    # Dataset
    datasets = {
        x: PIPAL(root_dir=Path(cfg.DATASETS.ROOT_DIR),
                 mode=x,
                 img_size=cfg.DATASETS.IMG_SIZE)
        for x in ['train', 'val', 'test']
    }

    datasets_size = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    # DataLoader
    dataloaders = {
        x: DataLoader(datasets[x],
                      batch_size=cfg.DATASETS.BATCH_SIZE,
                      shuffle=True,
                      num_workers=cfg.DATASETS.NUM_WORKERS)
        for x in ['train', 'val', 'test']
    }
    return dataloaders, datasets_size
