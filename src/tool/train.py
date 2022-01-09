import torch
from tqdm import tqdm

from src.tool.evaluate import calculate_correlation_coefficient


def train_phase2(dataloader, model, optimizer, loss, cfg, dataset_size, device=torch.device('cpu')):
    record = {
        'gt_scores': [],
        'pred_scores': []
    }

    result = {
        'real_loss': 0,
        'fake_loss': 0
    }

    with tqdm(dataloader) as tepoch:
        for iteration, (ref_imgs, dist_imgs, scores, categories, origin_scores) in enumerate(tepoch):
            ref_imgs = ref_imgs.to(device)
            dist_imgs = dist_imgs.to(device)
            scores = scores.to(device).float()
            categories = categories.to(device)

            # Format batch
            bs = ref_imgs.size(0)

            optimizer.zero_grad()

            """
            Deal with Real Distorted Images
            """
            _, _, pred_scores = model['netD'](ref_imgs, dist_imgs)

            real_loss = loss['mse_loss'](pred_scores, scores)

            # Record original scores and predict scores
            record['gt_scores'].append(origin_scores)
            record['pred_scores'].append(pred_scores.cpu().detach())

            """
            Deal with Fake Distorted Images
            """
            # Generate batch of latent vectors
            noise = torch.randn(bs, cfg.MODEL.LATENT_DIM, device=device)

            fake_imgs = model['netG'](ref_imgs,
                                      noise,
                                      scores.view(bs, -1),
                                      categories.view(bs, -1).float())

            _, _, pred_scores = model['netD'](ref_imgs, fake_imgs.detach())

            fake_loss = loss['mse_loss'](pred_scores, scores)

            total_loss = real_loss + fake_loss
            total_loss.backward()
            optimizer.step()

            result['real_loss'] += real_loss.item() * bs
            result['fake_loss'] += fake_loss.item() * bs

            # Show training message
            tepoch.set_postfix({
                'Real Loss': real_loss.item(),
                'Fake Loss': fake_loss.item(),
                'Total Loss': total_loss.item()
            })

    result['real_loss'] /= dataset_size
    result['fake_loss'] /= dataset_size

    """
    Calculate correlation coefficient
    """
    result['PLCC'], result['SRCC'], result['KRCC'] = \
        calculate_correlation_coefficient(
            torch.cat(record['gt_scores']).numpy(),
            torch.cat(record['pred_scores']).numpy()
        )

    return result
