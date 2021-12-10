import warnings

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm

warnings.simplefilter('ignore', np.RankWarning)


def calculate_correlation_coefficient(gt_qual, pred_qual):
    z = np.polyfit(pred_qual, gt_qual, 3)
    p = np.poly1d(z)

    return pearsonr(gt_qual, p(pred_qual))[0], \
           np.abs(spearmanr(gt_qual, pred_qual)[0]), \
           np.abs(kendalltau(gt_qual, pred_qual)[0])


def evaluate_phase1(dataloader, model, loss, dataset_size, device=torch.device('cpu')):
    record = {
        'gt_scores': [],
        'pred_scores': [],
    }

    result = {
        'real_clf': 0,
        'real_qual': 0,
    }

    model['netG'].eval()
    model['netD'].eval()

    for ref_imgs, dist_imgs, scores, categories, origin_scores in tqdm(dataloader):
        ref_imgs = ref_imgs.to(device)
        dist_imgs = dist_imgs.to(device)
        scores = scores.to(device).float()
        categories = categories.to(device)

        # Format batch
        bs = ref_imgs.size(0)

        with torch.no_grad():
            _, pred_categories, pred_scores = model['netD'](ref_imgs, dist_imgs)

            record['errD_real_clf'] = loss['ce_loss'](pred_categories, categories).item()
            record['errD_real_qual'] = loss['mse_loss'](pred_scores, scores).item()

        # Record original scores and predict scores
        record['gt_scores'].append(origin_scores)
        record['pred_scores'].append(pred_scores.cpu().detach())

        """
        Record epoch loss
        """
        result['real_clf'] += record['errD_real_clf'] * bs
        result['real_qual'] += record['errD_real_qual'] * bs

    result['real_clf'] /= dataset_size
    result['real_qual'] /= dataset_size

    """
    Calculate correlation coefficient
    """
    result['PLCC'], result['SRCC'], result['KRCC'] = \
        calculate_correlation_coefficient(
            torch.cat(record['gt_scores']).numpy(),
            torch.cat(record['pred_scores']).numpy()
        )

    return result
