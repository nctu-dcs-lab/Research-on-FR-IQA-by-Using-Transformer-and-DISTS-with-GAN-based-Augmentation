import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.modeling.module import Generator


def img_transform(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def transform(ref_img, dist_img_1, img_size=(192, 192)):
    ref_img = TF.center_crop(ref_img, img_size)
    dist_img_1 = TF.center_crop(dist_img_1, img_size)

    ref_img = TF.normalize(TF.to_tensor(ref_img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dist_img_1 = TF.normalize(TF.to_tensor(dist_img_1), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return ref_img, dist_img_1


def main():
    img_num = 1
    dist_class = (0, 2)
    assign_score = None
    device = torch.device("cpu")
    root_dir = Path('../data/PIPAL(processed)')
    latent_dim = 100
    label_dir = {'train': 'Train_Label', 'val': 'Val_Label', 'test': 'Test_Label'}
    dist_type = {
        '00': 0,
        '01': 12,
        '02': 12 + 16,
        '03': 12 + 16 + 10,
        '04': 12 + 16 + 10 + 24,
        '05': 12 + 16 + 10 + 24 + 13,
        '06': 12 + 16 + 10 + 24 + 13 + 14
    }

    netG = Generator().to(device)
    netG.load_state_dict(torch.load(args.netG_path))
    netG.eval().to(device)

    dfs = []
    for filename in (root_dir / label_dir['train']).glob('*.txt'):
        df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    df['category'] = df['dist_img'].apply(lambda x: dist_type[x[6:8]] + int(x[9:11]))
    df['ref_img'] = df['dist_img'].apply(lambda x: root_dir / f'Ref/{x[:5] + x[-4:]}')
    df['dist_img'] = df['dist_img'].apply(lambda x: root_dir / f'Dist/{x}')
    df['norm_score'] = 1 - ((df['score'] - np.min(df['score'])) / (np.max(df['score']) - np.min(df['score'])))

    df = df[['dist_img', 'ref_img', 'category', 'score', 'norm_score']].sort_values('dist_img')

    ref_img_path = f'../data/PIPAL(processed)/Ref/A{img_num:04d}.bmp'
    dist_img_path = f'../data/PIPAL(processed)/Dist/A{img_num:04d}_{dist_class[0]:02d}_{dist_class[1]:02d}.bmp'
    dist_cat = float(df[df['dist_img'] == Path(dist_img_path)]['category'])
    if assign_score:
        norm_score = assign_score
    else:
        norm_score = float(df[df['dist_img'] == Path(dist_img_path)]['norm_score'])

    print(f'Distorted Image Score: {1 - float(df[df["dist_img"] == Path(dist_img_path)]["norm_score"])}')
    print(f'Generated Image Score: {1 - norm_score}')

    ref_img = Image.open(ref_img_path).convert('RGB')
    dist_img = Image.open(dist_img_path).convert('RGB')

    ref_img, dist_img = transform(ref_img, dist_img)

    ref_img = ref_img.to(device)

    # Format batch
    c, h, w = ref_img.size()

    with torch.no_grad():
        # Generate batch of latent vectors
        noise = torch.randn(1, latent_dim, device=device)
        fake_img = netG(ref_img.view(-1, c, h, w),
                        noise,
                        torch.Tensor([[norm_score]]),
                        torch.Tensor([[dist_cat]]))

        fake_img = img_transform(fake_img[0].cpu().detach())

    ref_img = Image.fromarray(np.uint8(img_transform(ref_img) * 255), mode='RGB')
    dist_img = Image.fromarray(np.uint8(img_transform(dist_img) * 255), mode='RGB')
    fake_img = Image.fromarray(np.uint8(fake_img * 255), mode='RGB')

    ref_img.save('output_img/ref_img.bmp')
    dist_img.save('output_img/dist_img.bmp')
    fake_img.save('output_img/fake_img.bmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG_path', required=True, type=str, help='Load model path')
    args = parser.parse_args()

    main()
