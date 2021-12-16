from torch.utils.tensorboard import SummaryWriter


def write_iteration_log(writer: SummaryWriter, record, iteration, cfg):
    writer.add_scalars(
        'Loss_netD', {
            'total': record['errD'],
            'weighted_real_adv': cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_ADV * record['errD_real_adv'],
            'weighted_real_clf': cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_CLF * record['errD_real_clf'],
            'weighted_real_qual': cfg.TRAIN.CRITERION_WEIGHT.ERRD_REAL_QUAL * record['errD_real_qual'],
            'weighted_fake_adv': cfg.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_ADV * record['errD_fake_adv'],
            'weighted_fake_clf': cfg.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_CLF * record['errD_fake_clf']
        }, iteration // 100
    )

    writer.add_scalars(
        'Loss_netG', {
            'total': record['errG'],
            'weighted_adv': cfg.TRAIN.CRITERION_WEIGHT.ERRG_ADV * record['errG_adv'],
            'weighted_clf': cfg.TRAIN.CRITERION_WEIGHT.ERRG_CLF * record['errG_clf'],
            'weighted_qual': cfg.TRAIN.CRITERION_WEIGHT.ERRG_QUAL * record['errG_qual'],
            'weighted_cont': cfg.TRAIN.CRITERION_WEIGHT.ERRG_CONT * record['errG_cont']
        }, iteration // 100
    )

    writer.add_scalars(
        'Loss_adversarial', {
            'netD_real': record['errD_real_adv'],
            'netD_fake': record['errD_fake_adv'],
            'netG_fake': record['errG_adv']
        }, iteration // 100
    )

    writer.add_scalars(
        'Validity', {
            'D(x)': record['D_x'],
            'D(G(z))1': record['D_G_z1'],
            'D(G(z))2': record['D_G_z2']
        }, iteration // 100
    )

    writer.add_images(
        'Real Distorted Image',
        record['real_imgs'],
        iteration // 100,
        dataformats='NHWC'
    )
    writer.add_images(
        'Fake Distorted Image',
        record['fake_imgs'],
        iteration // 100,
        dataformats='NHWC'
    )
    writer.flush()


def write_epoch_log(writer: SummaryWriter, results, epoch):
    writer.add_scalars('Loss_classification',
                       {'train_real': results['train']['real_clf'],
                        'train_fake': results['train']['fake_clf'],
                        'val_real': results['val']['real_clf']},
                       epoch)
    writer.add_scalars('Loss_quality',
                       {'train_real': results['train']['real_qual'],
                        'train_fake': results['train']['fake_qual'],
                        'val_real': results['val']['real_qual']},
                       epoch)
    writer.add_scalars('PLCC', {x: results[x]['PLCC'] for x in ['train', 'val']}, epoch)
    writer.add_scalars('SRCC', {x: results[x]['SRCC'] for x in ['train', 'val']}, epoch)
    writer.add_scalars('KRCC', {x: results[x]['KRCC'] for x in ['train', 'val']}, epoch)
    writer.add_scalar('Loss_content',
                      results['train']['cont'],
                      epoch)
    writer.add_scalar('FID', results['train']['FID'], epoch)
    writer.flush()
