from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.data.dataset import PIPAL
from src.modeling.module import MultiTask

num_epoch = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 2e-4
batch_size = 16
num_workers = 6


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.regression = nn.Linear(2, 1)

    def forward(self, x):
        return self.regression(x)


# Dataset
datasets = {
    x: PIPAL(root_dir=Path('../data/PIPAL(processed)'),
             mode=x)
    for x in ['train', 'val', 'test']
}

datasets_size = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

# DataLoader
dataloaders = {
    x: DataLoader(datasets[x],
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=num_workers)
    for x in ['train', 'val', 'test']
}

low_level_cfg = get_cfg_defaults()
low_level_cfg.merge_from_file('src/config/experiments/1.1.7_config.yaml')
medium_level_cfg = get_cfg_defaults()
medium_level_cfg.merge_from_file('src/config/experiments/1.3.2_config.yaml')

low_level_cfg.freeze()
medium_level_cfg.freeze()

low_level_iqt = MultiTask(low_level_cfg).to(device)
low_level_iqt.load_state_dict(torch.load('experiments/1.1.7/models/netD_epoch79.pth'))
for parameter in low_level_iqt.parameters():
    parameter.requires_grad = False
low_level_iqt.eval()

medium_level_iqt = MultiTask(medium_level_cfg).to(device)
medium_level_iqt.load_state_dict((torch.load('experiments/1.3.2/models/netD_epoch57.pth')))
for parameter in medium_level_iqt.parameters():
    parameter.requires_grad = False
medium_level_iqt.eval()

regression = Regression().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(regression.parameters(), lr=lr)

for epoch in range(num_epoch):
    print(f'Epoch {epoch + 1}/{num_epoch}')
    print('-' * 10)

    for phase in ['train', 'val']:

        if phase == 'train':
            regression.train()
        else:
            regression.eval()

        running_loss = 0

        for ref_imgs, dist_imgs, scores, categories, origin_scores in tqdm(dataloaders[phase]):
            ref_imgs = ref_imgs.to(device)
            dist_imgs = dist_imgs.to(device)
            scores = scores.to(device).float()

            if phase == 'train':
                _, _, low_level_pred = low_level_iqt(ref_imgs, dist_imgs)
                _, _, medium_level_pred = medium_level_iqt(ref_imgs, dist_imgs)
            else:
                # Format batch
                bs, ncrops, c, h, w = ref_imgs.size()
                _, _, low_level_pred_scores = low_level_iqt(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
                low_level_pred = low_level_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)
                _, _, medium_level_pred_scores = medium_level_iqt(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
                medium_level_pred = medium_level_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

            with torch.set_grad_enabled(phase == 'train'):
                final_pred = regression(torch.cat((low_level_pred.view(-1, 1), medium_level_pred.view(-1, 1)), 1))

                loss = criterion(final_pred.view(-1), scores)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * ref_imgs.size(0)

        epoch_loss = running_loss / datasets_size[phase]

        print(f'{phase} Loss: {epoch_loss: .4f}')
