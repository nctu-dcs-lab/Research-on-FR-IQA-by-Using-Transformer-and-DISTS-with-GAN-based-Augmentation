import argparse
import os
import pickle

import pandas as pd
import scipy.io as sio
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.modeling.module import MultiTask


def transform(ref_img, dist_img, img_size=(192, 192)):
    ref_imgs = TF.five_crop(ref_img, img_size)
    dist_imgs = TF.five_crop(dist_img, img_size)

    ref_imgs = torch.stack([TF.normalize(TF.to_tensor(crop),
                                         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            for crop in ref_imgs])
    dist_imgs = torch.stack([TF.normalize(TF.to_tensor(crop),
                                          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             for crop in dist_imgs])

    return ref_imgs, dist_imgs


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = '../data/LIVE'

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))

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
    dmos = sio.loadmat(os.path.join(root_dir, 'dmos.mat'))['dmos']

    df = pd.DataFrame({'ref_img': refnames_all[0], 'dist_img': dist_path_list, 'dmos': dmos[0]})
    df['ref_img'] = df['ref_img'].apply(lambda x: os.path.join(root_dir, f'refimgs/{x[0]}'))
    df['dist_img'] = df['dist_img'].apply(lambda x: os.path.join(root_dir, f'{x}'))

    pred_scores_list = []

    netD.eval()
    for index, data in tqdm(df.iterrows()):
        ref_img = Image.open(data['ref_img']).convert('RGB')
        dist_img = Image.open(data['dist_img']).convert('RGB')
        ref_img, dist_img = transform(ref_img, dist_img)

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

    with open(args.output_file + '.pickle', 'wb') as handle:
        pickle.dump(pred_scores_arr, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Configuration YAML file for evaluating')
    parser.add_argument('--netD_path', required=True, type=str, help='Load model path')
    parser.add_argument('--output_file', default='output', type=str, help='Result file name')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    try:
        cfg.merge_from_file(args.config)
    except:
        print('Using default configuration file')

    assert cfg.MODEL.BACKBONE.NAME in ['InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT']

    if cfg.MODEL.BACKBONE.FEAT_LEVEL == 'low':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'medium':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(1088 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(10 * 10 for _ in range(6))
    elif cfg.MODEL.BACKBONE.FEAT_LEVEL == 'high':
        cfg.MODEL.BACKBONE.CHANNELS = tuple(2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(4 * 4 for _ in range(6))
    else:  # mixed
        cfg.MODEL.BACKBONE.CHANNELS = tuple(320 for _ in range(6)) + tuple(1088 for _ in range(6)) + tuple(
            2080 for _ in range(6))
        cfg.MODEL.BACKBONE.OUTPUT_SIZE = tuple(21 * 21 for _ in range(6)) + tuple(10 * 10 for _ in range(6)) + tuple(
            4 * 4 for _ in range(6))

    cfg.freeze()

    main(args, cfg)
