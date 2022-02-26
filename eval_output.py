import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.config.config import get_cfg_defaults
from src.modeling.module import MultiTask


def transform(ref_img, dist_img, img_size):
    ref_imgs = TF.five_crop(ref_img, img_size)
    dist_imgs = TF.five_crop(dist_img, img_size)

    ref_imgs = torch.stack([TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for crop in
                            ref_imgs])
    dist_imgs = torch.stack([TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for crop in
                             dist_imgs])

    return ref_imgs, dist_imgs


def main(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = Path(cfg.DATASETS.ROOT_DIR)
    img_size = cfg.DATASETS.IMG_SIZE
    label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}

    netD = MultiTask(cfg).to(device)
    netD.load_state_dict(torch.load(args.netD_path))

    records = {}

    for mode in ['val', 'test']:
        dfs = []
        for filename in (root_dir / label_dir[mode]).glob('*.txt'):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
        df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')
        df = df[['dist_img', 'ref_img']].sort_values('dist_img')

        pred_scores_list = []

        netD.eval()
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

        records[mode] = pred_scores_arr

    with open(args.output_file + '.pickle', 'wb') as handle:
        pickle.dump(records, handle)


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

    assert cfg.MODEL.BACKBONE.NAME in ['VGG16', 'InceptionResNetV2']
    assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed', 'reduced mixed']
    assert cfg.MODEL.EVALUATOR in ['IQT', 'DISTS', 'Transformer']

    cfg.freeze()

    main(args, cfg)
