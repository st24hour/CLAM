import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from datasets.dataset import Split_Gene_Clip_Dataset, Clip_dataset, save_clip_splits
from utils.core_utils_clip import train_wrapper

parser = argparse.ArgumentParser(description='CLIP training with WSI and genomics')
# deep learning hyperparam
parser.add_argument('--epochs', type=int, default=200, help='Number of train epochs')
parser.add_argument('--warmup_iter', type=int, default=2000, help='Number of warmup iterations')
parser.add_argument('--per_gpu_train_batch_size', type=int, default=8, help='The number of training batch size per GPU')
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8, help='The number of evaluation batch size per GPU')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--optimizer', type=str, choices = ['adam', 'adamw', 'sgd'], default='adam')
parser.add_argument('--decay_epoch', type=int, nargs='+', default=[150,225], help='Learning Rate Decay Steps. This parameter is not in use if the scheduler is set to "cosine".')
parser.add_argument('--scheduler', type=str, default='cosine', help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0005)')
parser.add_argument('--wd', type=float, default=0.2, help='weight decay (default: 0.2)')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')

# dataset hyperparam
parser.add_argument('--num_splits', type=int, default=5, help='Number of splits')
parser.add_argument('--wsi_csv_path', type=str, default='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv', help='WSI csv path')
parser.add_argument('--genomics_csv_path', type=str, default='/shared/js.yun/data/CLAM_data/genomics_data/TCGA-lung-LUAD+LUSC-selected_2847_zscore.csv', help='genomics csv path')
parser.add_argument('--wsi_feature_dir', type=str, default='/shared/j.jang/pathai/data/TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/', help='WSI pre-extracted feature path')
parser.add_argument('--split_dir', type=str, default='/shared/js.yun/data/CLAM_data/clip_data/', help='split csv location')
parser.add_argument('--save_dir', type=str, default='/shared/js.yun/logs/CLAM/temp/', help='directory to save logs')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation set fraction')
parser.add_argument('--test_frac', type=float, default=0.2, help='Test set fraction')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--dim_genom', type=int,  default=2847, help='Dimenstion of input genomics data vector')
# model hyperparam
parser.add_argument('--size_arg', type=str, choices=['small', 'big', 'custom1', 'HIPT_4k_feat', 'HIPT_256_feat', 'custom2_big'], default='custom2_big', help='size of model, does not affect mil')
# Misc
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()




device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

split_dataset = Split_Gene_Clip_Dataset(args.wsi_csv_path, args.genomics_csv_path, args.seed, args.num_splits, args.val_frac, args.test_frac)

args.split_dir=(f'{args.split_dir}TCGA-lung-splits_{args.num_splits}-frac_{1-args.val_frac-args.test_frac}_{args.val_frac}_{args.test_frac}-seed{args.seed}')
os.makedirs(args.split_dir, exist_ok=True)

if not os.path.isfile(os.path.join(args.split_dir, 'splits_0.csv')):
    for i in range(args.num_splits):
        train_index, val_index, test_index = next(split_dataset.create_splits_index())
        splits = split_dataset.create_split_file_name(train_index, val_index, test_index)

        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = ['train', 'val', 'test']
        save_clip_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))
        save_clip_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)

# logging folder setting
hp_setting = f'{args.optimizer}_lr_{args.lr}_sch_{args.scheduler}_decay_{",".join(str(i) for i in args.decay_epoch)}_wd_{args.wd}_epoch{args.epochs}_'.replace(" ", "")
i=0
while os.path.isdir(os.path.join(args.save_dir, hp_setting + f'_seed_{args.seed}_{str(i)}')):
    i+=1
args.save_dir = os.path.join(args.save_dir, hp_setting + f'_seed_{args.seed}_{str(i)}')
os.makedirs(args.save_dir, exist_ok=True)


# training 
folds = np.arange(args.num_splits)
for i in folds:
    split_csv_path=f'{args.split_dir}/splits_{i}.csv' 
    train_wrapper(split_csv_path, i, args)
