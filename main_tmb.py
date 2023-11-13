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
# from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Multi_Task_Dataset, save_splits
# from datasets.dataset_generic import save_splits

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
from datasets.dataset import Split_Dataset, Multi_Task_Dataset, save_splits


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

    results = []
    # all_test_auc = []
    # all_val_auc = []
    # all_test_acc = []
    # all_val_acc = []
    # if 'multi' in args.task and len(args.target_subtype)>=2:
    #     all_test_auc_LUSC = []
    #     all_test_auc_LUAD = []
    #     all_val_auc_LUSC = []
    #     all_val_auc_LUAD = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        split_path=f'{args.split_dir}/splits_{i}.csv' 
        # 공통 옵션 설정
        common_args = {
            "csv_path": args.csv_path,
            "split_path": split_path,
            "data_dir": os.path.join(args.data_root_dir, args.feature_folder),
            "shuffle": False,
            "seed": args.seed,
            "print_info": True,
            "label_dict": args.label_dict,
            "label_dict2": args.label_dict2,
            "use_h5": None,
            "target_subtype": args.target_subtype,
            "label_column": args.label_column,
            "tmb_threshold":args.tmb_threshold,
            "regression": args.regression,
            "balance": args.loss_balance
        }

        train_dataset = Multi_Task_Dataset(**common_args, split_key='train')
        val_dataset = Multi_Task_Dataset(**common_args, split_key='val')
        test_dataset = Multi_Task_Dataset(**common_args, split_key='test')
        datasets = (train_dataset, val_dataset, test_dataset)

        results_dict, output = train(datasets, i, args)      
        results.append({'folds': i, **output})
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results_dict)

    final_df = pd.DataFrame(results)

#########################
    #     result_list = train(datasets, i, args)

    #     all_test_auc.append(result_list[1])
    #     all_val_auc.append(result_list[2])
    #     all_test_acc.append(result_list[3])
    #     all_val_acc.append(result_list[4])
    #     if 'multi' in args.task and len(args.target_subtype)>=2:
    #         all_test_auc_LUSC.append(result_list[5])
    #         all_test_auc_LUAD.append(result_list[6])
    #         all_val_auc_LUSC.append(result_list[7])
    #         all_val_auc_LUAD.append(result_list[8])

    #     #write results to pkl
    #     filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
    #     save_pkl(filename, result_list[0])

    # final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
    #     'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})
    # if 'multi' in args.task and len(args.target_subtype)>=2:
    #     final_df['test_auc_LUSC'] = all_test_auc_LUSC
    #     final_df['test_auc_LUAD'] = all_test_auc_LUAD
    #     final_df['val_auc_LUSC'] = all_val_auc_LUSC
    #     final_df['val_auc_LUAD'] = all_val_auc_LUAD
##############################

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
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_sb_multi', 'clam_mb_multi'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'custom1', 'HIPT_4k_feat', 'HIPT_256_feat', 'custom2_big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'multi_task'])
parser.add_argument('--label_dict', type=json.loads, default='{"TMB_low":0, "TMB_high":1}')
parser.add_argument('--label_dict2', type=json.loads, default='{"LUSC":0, "LUAD":1}')
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
parser.add_argument('--loss_balance', type=float, nargs='+', default=[1.,0.,1.], help='weight term for balancing three losses')
parser.add_argument('--label_column', type=str, choices=['Mutation Count', 'TMB (nonsynonymous)', 'Subtype'], default='TMB (nonsynonymous)', help='tmb label column')
parser.add_argument('--target_subtype', type=str, nargs='+', default=['LUSC', 'LUAD'], help='tmb label column')
parser.add_argument('--tmb_high_ratio', type=float, default=0.3, help='tmb high percentage')
parser.add_argument('--val_frac', type=float, nargs='+', default=[0.1, 0.1], help='validation set ratio')
parser.add_argument('--test_frac', type=float, nargs='+', default=[0.2, 0.2], help='test set ratio')
parser.add_argument('--tmb_threshold', type=float, nargs='+', default=None, help='tmb threshold for each class. e.g. [323, 200]')
parser.add_argument('--regression', action='store_true', default=False, help='regression loss instead of classification')
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


# data split 폴더만 만들기 위한 데이터셋. 실제 데이터셋은 따로 만들 예정
args.n_classes = len(args.label_dict)
args.n_classes_subtype = len(args.label_dict2)
split_dataset = Split_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            label_dict_subtype = args.label_dict2,
                            patient_strat = True,
                            patient_voting='maj',
                            target_subtype=args.target_subtype,
                            label_column = args.label_column,
                            tmb_high_ratio = args.tmb_high_ratio,
                            balance = args.loss_balance
                            )

if args.loss_balance[0]:
    if args.label_column not in ['TMB (nonsynonymous)', 'Mutation Count']:
        raise ValueError('--label_column must be either "TMB (nonsynonymous)" or "Mutation Count" when --loss_balance[0] is True')
    elif args.tmb_threshold is None:
        args.tmb_threshold = split_dataset.tmb_threshold
    else:
        raise NotImplementedError('Please do not provide a value for args.threshold. The threshold value is adjusted solely through tmb_high_ratio.')
if args.loss_balance[1] and args.model_type == 'clam_mb_multi':
    raise NotImplementedError('Please set args.balance[1]=0 for clam_mb_multi model')
if args.loss_balance[2] and args.regression:
    raise ValueError("Instance loss is incompatible with regression. Please set args.loss_balance[2] to False.")

args.split_dir=(f'{args.split_dir}TCGA-lung-label_col_{args.label_column}_sub_{",".join(args.target_subtype)}'
                f'-TMB-high-ratio-{args.tmb_high_ratio:.2f}-splits_{args.k}-seed{args.seed}/{args.task}')
if not os.path.exists(os.path.join(args.split_dir, 'splits_0.csv')):
    num_slides_cls = np.array([len(cls_ids) for cls_ids in split_dataset.patient_cls_ids])     
    val_num = np.round(num_slides_cls * args.val_frac).astype(int)      # class별로 나눠서 비율 맞춰서 뽑음
    test_num = np.round(num_slides_cls * args.test_frac).astype(int)
    # print(val_num, test_num)
    # split_dir = args.split_dir + str(args.task) + '_{}'.format(int(1. * 100))
    os.makedirs(args.split_dir, exist_ok=True)
    # self.split_gen 생성. 호출할떄마다 yield sampled_train_ids, all_val_ids, all_test_ids
    split_dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=1.)  

    for i in range(args.k):
        # self.train_ids, self.val_ids, self.test_ids 생성
        split_dataset.set_splits()
        # train, val, test 각 class 별로 몇 개인지 excel file 저장
        descriptor_df = split_dataset.test_split_gen(return_descriptor=True)
        splits = split_dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))
        save_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
        descriptor_df.to_csv(os.path.join(args.split_dir, 'splits_{}_descriptor.csv'.format(i)))
    
def change_permissions_recursive(path, mode=0o777):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in [os.path.join(root,d) for d in dirs]:
            os.chmod(dir, mode)
    for file in [os.path.join(root, f) for f in files]:
            os.chmod(file, mode)


##################################################################
# 이하 전부 logging용 코드
args.scheduler = None
args.exp_code = f'{args.exp_code}_{args.model_type}_{",".join(args.target_subtype)}_{args.label_column}'.replace(" ", "")
if args.label_smoothing > 0:
    # args.exp_code = args.exp_code + '_LS'
    method = f'LS{args.label_smoothing}_'
elif args.focal_loss:
    # args.exp_code = args.exp_code + '_FL'
    method = f'FL_'
else:
    method = f''
hp_setting = (
    f'{method}ratio_{args.tmb_high_ratio:.2f}_{args.attn}_{args.opt}_lr_{args.lr}_sch_{args.scheduler}_'
    f'decay_{args.decay_epoch}_wd_{args.reg}_epoch{args.max_epochs}_'
    f'weighted_{"T" if args.weighted_sample else "F"}_drop_{args.drop_out}_'
    f'cluster_{"F" if args.no_inst_cluster else "T"}_B{args.B}'
).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")

i=0
while os.path.isdir(os.path.join(args.results_dir, str(args.exp_code) + '/' + hp_setting + f'_seed_{args.seed}_{str(i)}')):
    i+=1
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '/' + hp_setting + f'_seed_{args.seed}_{str(i)}')
os.makedirs(args.results_dir, exist_ok=True)
change_permissions_recursive(args.results_dir+'/../')

# if args.split_dir is None:
#     args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
#else:
#    args.split_dir = os.path.join('splits', args.split_dir)

# print('split_dir: ', args.split_dir)
# assert os.path.isdir(args.split_dir)

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

