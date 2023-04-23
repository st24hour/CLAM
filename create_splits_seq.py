import pdb
import os
# os.chdir('/shared/j.jang/pathai/CLAM')
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, default=None,
                    choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--n_class', type=int, default=3,
                    help='num class')
parser.add_argument('--csv_path', type=str, default='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung.csv',
                    help='csv file path')
parser.add_argument('--split_dir', type=str, default='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/',
                    help='split folder save path')

args = parser.parse_args()

# args.label_dict = {'normal_tissue':0, 'tumor_tissue':1}
###
# args.label_frac = 1.0
# args.task = 'task_2_tumor_subtyping'
args.label_dict = {'LUSC':0, 'LUAD':1}
#args.label_dict = {'BRCA_Basal':0, 'BRCA_Her2':1, 'BRCA_LumA':2, 'BRCA_LumB':3, 'BRCA_Normal':4, 'Others':5}
#args.label_dict = {'BRCA_Basal':0, 'BRCA_Her2':1, 'BRCA_LumA':2, 'BRCA_LumB':3, 'Others':4}
#args.label_dict = {'Metaplastic Breast Cancer':0, 'Breast Invasive Mixed Mucinous Carcinoma':1,'Breast Invasive Lobular Carcinoma':2,'Breast Invasive Ductal Carcinoma':3,'Breast Invasive Carcinoma (NOS)':4}
#args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1, 'Medullary Carcinoma':2, 'Metaplastic Carcinoma':3, 'Mixed Histology (NOS)':4, 'Mucinous Carcinoma':5, 'Other':6} 
#args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1, 'Medullary Carcinoma':2, 'Metaplastic Carcinoma':3, 'Mucinous Carcinoma':4} 
# args.label_dict ={'Infiltrating Ductal Carcinoma':0, 'Infiltrating Lobular Carcinoma':1} 

#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung.csv'
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast.csv'
#args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast-no_normal.csv'
# args.csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# n_class=2#6
# args.split_dir = '/shared/j.jang/pathai/data/TCGA-breast-splits-tumor-major-two/'
###

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    # args.n_classes=n_class
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = args.split_dir + str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



