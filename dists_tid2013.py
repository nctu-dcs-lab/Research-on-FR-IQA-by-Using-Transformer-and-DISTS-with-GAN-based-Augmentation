import os
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from DISTS_pytorch import DISTS
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient


class TID2013(Dataset):
    def __init__(self, root_dir, img_size=(192, 192)):
        df = pd.read_csv(os.path.join(root_dir, 'mos.csv'))

        df['ref_img'] = df['ref_img'].apply(lambda x: os.path.join(root_dir, 'reference_images', f'{x}'))
        df['dist_img'] = df['dist_img'].apply(lambda x: os.path.join(root_dir, 'distorted_images', f'{x}'))

        self.df = df
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_img = Image.open(self.df['ref_img'].iloc[idx]).convert('RGB')
        dist_img = Image.open(self.df['dist_img'].iloc[idx]).convert('RGB')

        ref_img, dist_img = self.transform(ref_img, dist_img)

        return ref_img, dist_img, self.df['mos'].iloc[idx]

    def transform(self, ref_img, dist_img):
        ref_img = TF.resize(ref_img, self.img_size)
        dist_img = TF.resize(dist_img, self.img_size)

        ref_img = TF.to_tensor(ref_img)
        dist_img = TF.to_tensor(dist_img)

        return ref_img, dist_img


root_dir = Path('../data/TID2013')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D = DISTS().to(device)

dataset = TID2013(root_dir=root_dir, img_size=(256, 256))
dataloader = DataLoader(dataset,
                        batch_size=16,
                        shuffle=False,
                        num_workers=6)

record_pred_scores = []
record_gt_scores = []

for ref_imgs, dist_imgs, scores in tqdm(dataloader):
    ref_imgs = ref_imgs.to(device)
    dist_imgs = dist_imgs.to(device)

    pred_scores = D(ref_imgs, dist_imgs)
    record_pred_scores.append(pred_scores.cpu().detach())
    record_gt_scores.append(scores)

plcc, srcc, krcc = calculate_correlation_coefficient(
    torch.cat(record_gt_scores).numpy(),
    torch.cat(record_pred_scores).numpy()
)

print(f'TID2013 Dataset')
print(f'PLCC: {plcc}')
print(f'SRCC: {srcc}')
print(f'KRCC: {krcc}')
