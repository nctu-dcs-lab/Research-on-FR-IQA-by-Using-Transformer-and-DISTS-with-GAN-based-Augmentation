from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.data.dataset import PIPAL
from src.modeling.module import MultiTask
from src.tool.evaluate import calculate_correlation_coefficient

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_workers = 6

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

dists_cfg = get_cfg_defaults()
dists_cfg.merge_from_file('src/config/experiments/4.0.0_config.yaml')

low_level_cfg.freeze()
dists_cfg.freeze()

low_level_iqt = MultiTask(low_level_cfg).to(device)
low_level_iqt.load_state_dict(torch.load('experiments/1.1.7/models/netD_epoch79.pth'))
for parameter in low_level_iqt.parameters():
    parameter.requires_grad = False
low_level_iqt.eval()

dists = MultiTask(dists_cfg).to(device)
dists.load_state_dict(torch.load('experiments/4.0.0/models/netD_epoch113.pth'))
for parameter in dists.parameters():
    parameter.requires_grad = False
dists.eval()

criterion = nn.MSELoss()

running_loss = 0
gt_scores = []
pred_scores = []

for ref_imgs, dist_imgs, scores, categories, origin_scores in tqdm(dataloaders['val']):
    ref_imgs = ref_imgs.to(device)
    dist_imgs = dist_imgs.to(device)
    scores = scores.to(device).float()

    # Format batch
    bs, ncrops, c, h, w = ref_imgs.size()

    _, _, low_level_pred_scores = low_level_iqt(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
    low_level_pred = low_level_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

    _, _, dists_pred_scores = dists(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
    dists_pred = dists_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

    final_pred = (low_level_pred + dists_pred) / 2

    loss = criterion(final_pred.view(-1), scores)

    # statistics
    running_loss += loss.item() * ref_imgs.size(0)
    gt_scores.append(origin_scores)
    pred_scores.append(final_pred.cpu().detach())

epoch_loss = running_loss / datasets_size['val']

plcc, srcc, krcc = calculate_correlation_coefficient(
    torch.cat(gt_scores).numpy(),
    torch.cat(pred_scores).numpy()
)

print(f'val Loss: {epoch_loss}')
print(f'val PLCC: {plcc}, SRCC: {srcc}, KRCC: {krcc}')

gt_scores = []
pred_scores = []

for ref_imgs, dist_imgs, _, _, origin_scores in tqdm(dataloaders['test']):
    ref_imgs = ref_imgs.to(device)
    dist_imgs = dist_imgs.to(device)

    # Format batch
    bs, ncrops, c, h, w = ref_imgs.size()

    _, _, low_level_pred_scores = low_level_iqt(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
    low_level_pred = low_level_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

    _, _, dists_pred_scores = dists(ref_imgs.view(-1, c, h, w), dist_imgs.view(-1, c, h, w))
    dists_pred = dists_pred_scores.view(bs, ncrops, -1).mean(1).view(-1)

    final_pred = (low_level_pred + dists_pred) / 2

    gt_scores.append(origin_scores)
    pred_scores.append(final_pred.cpu().detach())

plcc, srcc, krcc = calculate_correlation_coefficient(
    torch.cat(gt_scores).numpy(),
    torch.cat(pred_scores).numpy()
)

print(f'test PLCC: {plcc}, SRCC: {srcc}, KRCC: {krcc}')
