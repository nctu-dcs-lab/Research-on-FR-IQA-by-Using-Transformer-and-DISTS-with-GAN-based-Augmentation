import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import scipy.io as sio
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.modeling.module import MultiTask


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


def get_LIVE_df():
    root_dir = '../data/LIVE'
    num_type_map = {
        'jp2k': 227,
        'jpeg': 233,
        'wn': 174,
        'gblur': 174,
        'fastfading': 174
    }

    dist_path_list = []
    for dist_type, num_dist in num_type_map.items():
        for i in range(1, num_dist + 1):
            dist_path_list.append(os.path.join(dist_type, f'img{i}.bmp'))

    refnames_all = sio.loadmat(os.path.join(root_dir, 'refnames_all.mat'))['refnames_all']

    df = pd.DataFrame({'ref_img': refnames_all[0], 'dist_img': dist_path_list})
    df['ref_img'] = df['ref_img'].apply(lambda x: os.path.join(root_dir, f'refimgs/{x[0]}'))
    df['dist_img'] = df['dist_img'].apply(lambda x: os.path.join(root_dir, f'{x}'))

    return df


def get_TID2013_df():
    root_dir = '../data/TID2013'
    df = pd.read_csv(os.path.join(root_dir, 'mos.csv'))

    df['ref_img'] = df['ref_img'].apply(lambda x: os.path.join(root_dir, 'reference_images', f'{x}'))
    df['dist_img'] = df['dist_img'].apply(lambda x: os.path.join(root_dir, 'distorted_images', f'{x}'))

    return df


def get_pred_scores(df, netD, img_size, device):
    def transform(ref_img, dist_img, img_size):
        ref_imgs = TF.five_crop(ref_img, img_size)
        dist_imgs = TF.five_crop(dist_img, img_size)

        ref_imgs = torch.stack([TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for crop
                                in
                                ref_imgs])
        dist_imgs = torch.stack([TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for crop
                                 in
                                 dist_imgs])

        return ref_imgs, dist_imgs

    pred_scores_list = []

    for index, data in tqdm(df.iterrows()):
        ref_img = Image.open(data['ref_img']).convert('RGB')
        dist_img = Image.open(data['dist_img']).convert('RGB')
        ref_img, dist_img = transform(ref_img, dist_img, img_size)

        ref_img = ref_img.to(device)
        dist_img = dist_img.to(device)

        # Format batch
        ncrops, c, h, w = ref_img.size()

        with torch.no_grad():
            """
            Evaluate distorted images
            """
            _, _, pred_scores = netD(ref_img.view(-1, c, h, w), dist_img.view(-1, c, h, w))
            pred_scores_avg = pred_scores.view(1, ncrops, -1).mean(1).view(-1)

            # Record original predict scores
            pred_scores_list.append(pred_scores_avg.cpu().detach())

    pred_scores_arr = torch.cat(pred_scores_list).numpy()

    return pred_scores_arr


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = cfg.DATASETS.IMG_SIZE

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))
    netD.eval()

    if args.dataset == 'PIPAL':
        records = {}
        for dataset_type in ['train', 'val', 'test']:
            df = get_PIPAL_df(dataset_type)
            records[dataset_type] = get_pred_scores(df, netD, img_size, device)

    elif args.dataset == 'LIVE':
        df = get_LIVE_df()
        records = get_pred_scores(df, netD, img_size, device)

    else:
        df = get_TID2013_df()
        records = get_pred_scores(df, netD, img_size, device)

    with open(args.output, 'wb') as handle:
        pickle.dump(records, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for evaluating')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    parser.add_argument('--output', default='pred_scores.pickle', type=str, help='Output file name of a pickle file')
    parser.add_argument('--dataset',
                        default='PIPAL',
                        choices=['PIPAL', 'LIVE', 'TID2013'],
                        help='Dataset to be evaluated')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')

    assert cfg.MODEL.BACKBONE.NAME in ['VGG16', 'InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed', 'reduced mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    cfg.freeze()

    main(args, cfg)
