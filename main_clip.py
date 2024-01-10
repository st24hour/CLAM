import os
import argparse
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from datasets.dataset import Split_Gene_Clip_Dataset, Clip_dataset, save_clip_splits
from utils.core_utils_clip import train_wrapper
from utils.utils import seed_torch


parser = argparse.ArgumentParser(description='CLIP training with WSI and genomics')
# deep learning hyperparam
parser.add_argument('--epochs', type=int, default=20, help='Number of train epochs')
parser.add_argument('--warmup_iter', type=int, default=0, help='Number of warmup iterations')
# parser.add_argument('--per_gpu_train_batch_size', type=int, default=1, help='The number of training batch size per GPU')
# parser.add_argument('--per_gpu_eval_batch_size', type=int, default=1, help='The number of evaluation batch size per GPU')
# parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
parser.add_argument('--optimizer', type=str, choices = ['adam', 'adamw', 'sgd'], default='adamw')
parser.add_argument('--decay_epoch', type=int, nargs='+', default=[150,225], help='Learning Rate Decay Steps. This parameter is not in use if the scheduler is set to "cosine".')
parser.add_argument('--scheduler', type=str, choices = ['cos', 'None'], default='cos', help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0005)')
parser.add_argument('--wd', type=float, default=0.2, help='weight decay (default: 0.2)')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
# dataset hyperparam
parser.add_argument('--folds', type=int, default=1, help='Number of splits')
parser.add_argument('--wsi_csv_path', type=str, default='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv', help='WSI csv path')
parser.add_argument('--genomics_csv_path', type=str, default='/shared/js.yun/data/CLAM_data/genomics_data/TCGA-lung-LUAD+LUSC-selected_2847_zscore.csv', help='genomics csv path')
parser.add_argument('--wsi_feature_dir', type=str, default='/shared/j.jang/pathai/data/TCGA-lung-x256-features-dino-from-pretrained-vitb-ep9-img224/', help='WSI pre-extracted feature path')
parser.add_argument('--split_dir', type=str, default='/shared/js.yun/data/CLAM_data/clip_data/', help='split csv location')
parser.add_argument('--save_dir', type=str, default='/shared/js.yun/logs/CLAM/clip/base/', help='directory to save logs')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation set fraction')
parser.add_argument('--test_frac', type=float, default=0.2, help='Test set fraction')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
parser.add_argument('--dim_genom', type=int,  default=2847, help='Dimenstion of input genomics data vector')
# model hyperparam
parser.add_argument('--model', type=str, choices=['make_clip'], default='make_clip', help='Name of clip model function')
parser.add_argument('--size_arg', type=str, choices=['small', 'big', 'custom1', 'HIPT_4k_feat', 'HIPT_256_feat', 'custom2_big'], default='custom2_big', help='size of model, does not affect mil')
parser.add_argument('--image_encoder', type=str, choices=['CLAM_Encoder'], default='CLAM_Encoder', help='architecture of image encoder')
parser.add_argument('--genomics_encoder', type=str, choices=['Genomics_Encoder_FC', 'Genomics_Encoder_FC_Skip_early_v1', 'Identity'], default='fc', help='architecture of genomics encoder')
parser.add_argument('--output_dim', type=int,  default=None, help='Dimenstion of wsi, genomics embedding vector')
parser.add_argument('--hidden_dim_genom', type=int,  default=4096, help='Dimenstion of hidden layer of genomics_encoder_fc')
parser.add_argument('--num_layers_genom', type=int,  default=2, help='How many hidden layers of genomics_encoder_fc?, >=0')
parser.add_argument('--activation_func_genom', type=str,  default='ReLU', help='Activation function of Genomics encoder')
parser.add_argument('--norm_genom', type=str, choices=[None, 'LayerNorm'], default=None, help='normalization layer of genomics encoder')
# Misc
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--no_tensorboard', action='store_true', default=False, help='If True, tensorboard is not used')
parser.add_argument('--note', type=str, default=None, help='Extra logging note. e.g. model architecur')
parser.add_argument('--use_batch', action='store_true', default=False, help='If True, batch dimensoion is used.')
args = parser.parse_args()
args.tensorboard = not args.no_tensorboard

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(args.seed)


split_dataset = Split_Gene_Clip_Dataset(args.wsi_csv_path, args.genomics_csv_path, args.seed, args.folds, args.val_frac, args.test_frac)

args.split_dir=(f'{args.split_dir}TCGA-lung-splits_{args.folds}-frac_{1-args.val_frac-args.test_frac}_{args.val_frac}_{args.test_frac}-seed{args.seed}')
os.makedirs(args.split_dir, exist_ok=True)

if not os.path.isfile(os.path.join(args.split_dir, 'splits_0.csv')):
    for i in range(args.folds):
        train_index, val_index, test_index = next(split_dataset.create_splits_index())
        splits = split_dataset.create_split_file_name(train_index, val_index, test_index)

        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = ['train', 'val', 'test']
        save_clip_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))
        save_clip_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)

# logging folder setting
hp_setting = (
    f'{args.optimizer}_lr_{args.lr}_{args.scheduler}_{f"warm_{args.warmup_iter}" if args.warmup_iter else ""}'
    f'_b_{args.batch_size}{"_decay_"+",".join(str(i) for i in args.decay_epoch) if args.scheduler is "step" else ""}'
    f'_wd_{args.wd}_epoch_{args.epochs}{f"_{args.note}" if args.note is not None else ""}'
).replace(" ", "")
i=0
while os.path.isdir(os.path.join(args.save_dir, hp_setting + f'_seed_{args.seed}_{str(i)}')):
    i+=1
args.save_dir = os.path.join(args.save_dir, hp_setting + f'_seed_{args.seed}_{str(i)}')
os.makedirs(args.save_dir, exist_ok=True)


# training 
result_df = pd.DataFrame()
folds = np.arange(args.folds)
for i in folds:
    split_csv_path=f'{args.split_dir}/splits_{i}.csv' 
    output_dict = train_wrapper(args, split_csv_path, i)
    
    # save result
    temp_df = pd.DataFrame(output_dict, index=[i])
    result_df = pd.concat([result_df, temp_df], axis=0)

kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst).strftime("%y%m%d-%H%M%S")
file_name = f'results_{now}.csv'
result_df.to_csv(os.path.join(args.save_dir, file_name))