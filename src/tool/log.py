import wandb
from torchvision.utils import make_grid


def write_iteration_log(record, iteration, criterion_weight):
    real_imgs = wandb.Image(make_grid(record['real_imgs']))
    fake_imgs = wandb.Image(make_grid(record['fake_imgs']))

    wandb.log({
        'hundred iteration': iteration // 100,

        'netD/total_loss': record['errD'],
        'netD/weighted_real_adv_loss': criterion_weight['ERRD_REAL_ADV'] * record['errD_real_adv'],
        'netD/weighted_real_clf_loss': criterion_weight['ERRD_REAL_CLF'] * record['errD_real_clf'],
        'netD/weighted_real_qual_loss': criterion_weight['ERRD_REAL_QUAL'] * record['errD_real_qual'],
        'netD/weighted_fake_adv_loss': criterion_weight['ERRD_FAKE_ADV'] * record['errD_fake_adv'],
        'netD/weighted_fake_clf_loss': criterion_weight['ERRD_FAKE_CLF'] * record['errD_fake_clf'],
        'netD/real_adv_loss': record['errD_real_adv'],
        'netD/fake_adv_loss': record['errD_fake_adv'],

        'netG/total_loss': record['errG'],
        'netG/weighted_adv_loss': criterion_weight['ERRG_ADV'] * record['errG_adv'],
        'netG/weighted_clf_loss': criterion_weight['ERRG_CLF'] * record['errG_clf'],
        'netG/weighted_qual_loss': criterion_weight['ERRG_QUAL'] * record['errG_qual'],
        'netG/weighted_cont_loss': criterion_weight['ERRG_CONT'] * record['errG_cont'],
        'netG/adv_loss': record['errG_adv'],

        'Validity/D(x)': record['D_x'],
        'Validity/D(G(z))1': record['D_G_z1'],
        'Validity/D(G(z))2': record['D_G_z2'],

        'Image/real_imgs': real_imgs,
        'Image/fake_imgs': fake_imgs
    })


def write_epoch_log(results, epoch, phase=0):
    if phase == 1:
        wandb.log({
            'epoch': epoch,
            # record classification loss
            'train/real_classification_loss': results['train']['real_clf'],
            'train/fake_classification_loss': results['train']['fake_clf'],
            'val/real_classification_loss': results['val']['real_clf'],
            'val/fake_classification_loss': results['val']['fake_clf'],
            # record quality loss
            'train/real_quality_loss': results['train']['real_qual'],
            'train/fake_quality_loss': results['train']['fake_qual'],
            'val/real_quality_loss': results['val']['real_qual'],
            'val/fake_quality_loss': results['val']['fake_qual'],
            # record correlation coefficient
            'train/PLCC': results['train']['PLCC'],
            'val/PLCC': results['val']['PLCC'],
            'train/SRCC': results['train']['SRCC'],
            'val/SRCC': results['val']['SRCC'],
            'train/KRCC': results['train']['KRCC'],
            'val/KRCC': results['val']['KRCC'],
            # record generated image quality
            'train/content_loss': results['train']['cont'],
            'train/FID': results['train']['FID']
        })
    elif phase == 2:
        wandb.log({
            'epoch': epoch,
            # record quality loss
            'train/real_quality_loss': results['train']['real_qual'],
            'train/fake_quality_loss': results['train']['fake_qual'],
            'val/real_quality_loss': results['val']['real_qual'],
            'val/fake_quality_loss': results['val']['fake_qual'],
            # record correlation coefficient
            'train/PLCC': results['train']['PLCC'],
            'val/PLCC': results['val']['PLCC'],
            'train/SRCC': results['train']['SRCC'],
            'val/SRCC': results['val']['SRCC'],
            'train/KRCC': results['train']['KRCC'],
            'val/KRCC': results['val']['KRCC'],
        })
    else:
        wandb.log({
            'epoch': epoch,
            # record quality loss
            'train/real_quality_loss': results['train']['real_qual'],
            'val/real_quality_loss': results['val']['real_qual'],
            # record correlation coefficient
            'train/PLCC': results['train']['PLCC'],
            'val/PLCC': results['val']['PLCC'],
            'train/SRCC': results['train']['SRCC'],
            'val/SRCC': results['val']['SRCC'],
            'train/KRCC': results['train']['KRCC'],
            'val/KRCC': results['val']['KRCC'],
        })
