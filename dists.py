from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from DISTS_pytorch import DISTS
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient

root_dir = Path('../data/PIPAL(processed)')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D = DISTS().to(device)


class PIPAL(Dataset):
    def __init__(self, root_dir, dataset_type='train', img_size=(256, 256)):
        label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}

        dfs = []
        for filename in (root_dir / label_dir[dataset_type]).glob('*.txt'):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
        df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')

        self.origin_scores = df['score'].to_numpy()

        self.df = df[['dist_img', 'ref_img']]

        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_img = Image.open(self.df['ref_img'].iloc[idx]).convert('RGB')
        dist_img = Image.open(self.df['dist_img'].iloc[idx]).convert('RGB')

        ref_img, dist_img = self.transform(ref_img, dist_img)

        return ref_img, dist_img, self.origin_scores[idx]

    def transform(self, ref_img, dist_img):
        ref_imgs = TF.five_crop(ref_img, self.img_size)
        dist_imgs = TF.five_crop(dist_img, self.img_size)

        ref_imgs = torch.stack([TF.to_tensor(crop) for crop in ref_imgs])
        dist_imgs = torch.stack([TF.to_tensor(crop) for crop in dist_imgs])

        return ref_imgs, dist_imgs


datasets = {dataset_type: PIPAL(root_dir=root_dir, dataset_type=dataset_type)
            for dataset_type in ['train', 'val', 'test']}
datasets_size = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

# DataLoader
dataloaders = {dataset_type: DataLoader(datasets[dataset_type],
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=6) for dataset_type in ['train', 'val', 'test']}

record_pred_scores = []
record_gt_scores = []

for ref_imgs, dist_imgs, scores in tqdm(dataloaders['train']):
    ref_imgs = ref_imgs[0].to(device)
    dist_imgs = dist_imgs[0].to(device)

    pred_scores = D(ref_imgs, dist_imgs)
    avg_pred_scores = torch.mean(pred_scores)

    record_pred_scores.append(avg_pred_scores.cpu().detach().view(1))
    record_gt_scores.append(scores)

plcc, srcc, krcc = calculate_correlation_coefficient(
    torch.cat(record_gt_scores).numpy(),
    torch.cat(record_pred_scores).numpy()
)

print(plcc, srcc, krcc)
