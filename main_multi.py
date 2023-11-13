from __future__ import print_function

import argparse
import pdb
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="8, 9, 10, 11, 12, 13, 14, 15" ## must declare before importing torch
# os.environ["CUDA_VISIBLE_DEVICES"]="0" ## must declare before importing torch
# os.chdir('/shared/j.jang/pathai/CLAM')
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Multi_Task_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# js
import json
import logging


def main(args):
    # create results directory if necessary 
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1: #which fold start from?
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:    #which fold end?
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    if 'multi' in args.task:
        all_test_auc_LUSC = []
        all_test_auc_LUAD = []
        all_val_auc_LUSC = []
        all_val_auc_LUAD = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        if args.task == 'multi_task':       # Multi_Task_Dataset에는 from_id 부분을 없앰 
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                    csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        else:    
            train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                    csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        # all_test_auc.append(test_auc)
        # all_val_auc.append(val_auc)
        # all_test_acc.append(test_acc)
        # all_val_acc.append(val_acc)
        result_list  = train(datasets, i, args)
        # results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_auc_LUSC, test_auc_LUAD, val_auc_LUSC, val_auc_LUAD
        all_test_auc.append(result_list[1])
        all_val_auc.append(result_list[2])
        all_test_acc.append(result_list[3])
        all_val_acc.append(result_list[4])
        if 'multi' in args.task:
            all_test_auc_LUSC.append(result_list[5])
            all_test_auc_LUAD.append(result_list[6])
            all_val_auc_LUSC.append(result_list[7])
            all_val_auc_LUAD.append(result_list[8])

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, result_list[0])

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})
    if 'multi' in args.task:
        final_df['test_auc_LUSC'] = all_test_auc_LUSC
        final_df['test_auc_LUAD'] = all_test_auc_LUAD
        final_df['val_auc_LUSC'] = all_val_auc_LUSC
        final_df['val_auc_LUAD'] = all_val_auc_LUAD

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--decay_epoch', type=int, nargs='+', default=[150,225], 
                        help='Learning Rate Decay Steps')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--feature_folder', default='TCGA-lung-features', help='feature directory (default: TCGA-lung-features)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--csv_path', default='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv', help='csv file path')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'adamw', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_sb_multi'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'custom1', 'HIPT_4k_feat', 'HIPT_256_feat', 'custom2_big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'multi_task'])
parser.add_argument('--label_dict', type=json.loads, default='{"LUSC":0, "LUAD":1}')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--attn', type=str, default="gated", help='type of attention')
### JS MISC
parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing factor range 0~1')
parser.add_argument('--focal_loss', action='store_true', default=False, help='using focal loss')
parser.add_argument('--loss_balance', type=float, default=1., help='weight term for balancing two loss')
args = parser.parse_args()
# print(args.label_dict)
# print(type(args.label_dict))
# exit()

### TCGA-kidney subtyping ###
# args.bag_weight = 0.3         # 왜 0.7이 아니고 0.3으로 하셨는지?
# args.drop_out = True
# args.early_stopping = True
#args.lr = 2e-4
# args.lr = 5e-5                # 바꾸신 이유는?
# args.k = 10
# args.label_frac = 1.0

#args.exp_code = 'LUSC_vs_LUAD_CLAM_100'
#args.exp_code = 'task_2_tumor_subtyping_CLAM_100-ViT-B-16'
#args.exp_code = 'task_2_tumor_subtyping_CLAM_100'
#args.exp_code = 'task_2_tumor_cancer_typing_CLAM_100-ViT-L-14'
# args.exp_code = 'task_2_tumor_typing_only_major_two_CLAM_100'

# args.weighted_sample = True
# args.bag_loss = 'ce'
# args.inst_loss = 'svm'
#args.task = 'task_1_tumor_vs_normal'
# args.task = 'task_2_tumor_subtyping'
# args.model_type = 'clam_sb'
# args.log_data = True
# args.subtyping = True
# args.data_root_dir = '/shared/j.jang/pathai/data/'

#args.feature_folder = 'TCGA-kidney-features-VITB-16'
#args.feature_folder = 'TCGA-breast-features'
# args.feature_folder = 'TCGA-breast-features'

#args.results_dir = './TCGA-kidney-results/'
#args.results_dir = './TCGA-lung-results/'
# args.results_dir = './TCGA-breast-results/'

#args.split_dir = '/shared/j.jang/pathai/data/TCGA-kidney-splits/task_2_tumor_subtyping_100/'
#args.split_dir = '/shared/j.jang/pathai/data/TCGA-lung-splits/task_1_tumor_vs_normal_100/'
# args.split_dir = '/shared/j.jang/pathai/data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'

# args.label_dict = {'LUSC':0, 'LUAD':1}
#args.label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2}
#args.label_dict = {'BRCA_Basal':0, 'BRCA_Her2':1, 'BRCA_LumA':2, 'BRCA_LumB':3, 'Others':4}
#args.label_dict = {'Metaplastic Breast Cancer':0, 'Breast Invasive Mixed Mucinous Carcinoma':1,'Breast Invasive Lobular Carcinoma':2,'Breast Invasive Ductal Carcinoma':3,'Breast Invasive Carcinoma (NOS)':4}
#args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1, 'Medullary Carcinoma':2, 'Metaplastic Carcinoma':3, 'Mixed Histology (NOS)':4, 'Mucinous Carcinoma':5, 'Other':6} 
#args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1, 'Medullary Carcinoma':2, 'Metaplastic Carcinoma':3, 'Mucinous Carcinoma':4} 
# args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1}

n_classes = 2
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung.csv'
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-kidney.csv'
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast-cancer-type-label.csv'
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast-tumor-type-label.csv'
# args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'

#args.model_size = 'custom1' #for ViTB-16
#args.model_size = 'custom2' #for ViTL-14
# args.model_size = 'small'
###



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
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

# encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

args.n_classes=n_classes
if args.task == 'task_1_tumor_vs_normal':
    dataset = Generic_MIL_Dataset(#csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, args.feature_folder),
                            # data_dir= os.path.join(args.data_root_dir, 'TCGA-lung-features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            #label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            label_dict = args.label_dict,
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    #dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
    dataset = Generic_MIL_Dataset(csv_path=args.csv_path,
                            #data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            data_dir=os.path.join(args.data_root_dir, args.feature_folder),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            #label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            label_dict = args.label_dict,
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 

elif args.task == 'multi_task':
    dataset = Multi_Task_Dataset(csv_path=args.csv_path,
                            data_dir=os.path.join(args.data_root_dir, args.feature_folder),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            label_dict2 = {'LUSC':0, 'LUAD':1},
                            patient_strat= False,
                            ignore=[])
else:
    raise NotImplementedError
    
def change_permissions_recursive(path, mode=0o777):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in [os.path.join(root,d) for d in dirs]:
            os.chmod(dir, mode)
    for file in [os.path.join(root, f) for f in files]:
            os.chmod(file, mode)

# if not os.path.isdir(args.results_dir):
#     os.mkdir(args.results_dir)
# change_permissions_recursive(args.results_dir)

##################################################################
# 이하 전부 logging용 코드
args.scheduler = None
if args.label_smoothing > 0:
    args.exp_code = args.exp_code + '_LS'
    method = f'LS{args.label_smoothing}'
elif args.focal_loss:
    args.exp_code = args.exp_code + '_FL'
    method = f'FL'
else:
    method = f''
hp_setting = (
    f'{method}_{args.attn}_{args.opt}_lr_{args.lr}_sch_{args.scheduler}_'
    f'decay_{args.decay_epoch}_wd_{args.reg}_epoch{args.max_epochs}_'
    f'weighted_{"T" if args.weighted_sample else "F"}_drop_{args.drop_out}_'
    f'cluster_{"F" if args.no_inst_cluster else "T"}_B{args.B}'
).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")

i=0
while os.path.isdir(os.path.join(args.results_dir, str(args.exp_code) + '/' + hp_setting + f'_s{args.seed}_{str(i)}')):
    i+=1
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '/' + hp_setting + f'_s{args.seed}_{str(i)}')
os.makedirs(args.results_dir, exist_ok=True)
change_permissions_recursive(args.results_dir+'/../')

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
#else:
#    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()
with open(args.results_dir + '/args_{}.txt'.format(args.exp_code), 'w') as f:
    print(args, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    change_permissions_recursive(args.results_dir+'/../../')
    print("finished!")
    print("end script")

