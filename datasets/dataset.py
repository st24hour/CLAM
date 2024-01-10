from torchvision import transforms
# import openslide
import h5py
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
# from PIL import Image
import torch

from openslide.lowlevel import *
from scipy import stats
from utils.utils import generate_split, nth
import os
import random

pd.set_option('display.max_rows', None)  # 모든 행 표시
pd.set_option('display.max_columns', None)  # 모든 열 표시

'''
데이터셋 만들때마다 data folder에 split.csv 저장
동일 seed에서 split 똑같도록
'''

# Multi_Task_Dataset에서 안쓰면 Split_Dataset 안으로 넣을 수도 있음
def top_n_percent_values(slide_data, target_subtype, label_column, tmb_high_ratio):
    if not 0 <= tmb_high_ratio <= 1:
        raise ValueError("tmb_high_ratio should be between 0 and 1")

    tmb_threshold = []
    for one_subtype in target_subtype:
        if one_subtype not in slide_data['Subtype'].tolist():
            raise ValueError(f"'{one_subtype}' column is not in csv file")
        slide_data_subset = slide_data[slide_data['Subtype'] == one_subtype]
        tmb_values_list = slide_data_subset[label_column].tolist()
        tmb_values_list.sort(reverse=True)
        k = int(len(tmb_values_list) * tmb_high_ratio)  # 리스트 길이* n에 해당하는 인덱스를 계산. np.round 적용? 
        # print(len(tmb_values_list), k)
        # 상위 n 값들을 반환합니다.
        tmb_threshold.append(tmb_values_list[k-1])
        print(f'TMB high threshold for {one_subtype}: {tmb_values_list[k-1]}')
    return tmb_threshold


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    ##########################################################################################
    splits = [split_datasets[i]['case_id']+'/'+split_datasets[i]['slide_id'] for i in range(len(split_datasets))]
    # splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]		# HIPT data만 쓰는 경우 
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
       
       
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)


def save_clip_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i]['case_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
       
       
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)

class Split_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False,            # 여기서는 필요없음
        seed = 7, 
        print_info = True,
        label_dict = {},                    # {"TMB_low":0, "TMB_high":1}
        label_dict_subtype = {},            # {"LUSC":0, "LUAD":1}
        filter_dict = {},
        patient_strat=False,
        patient_voting = 'max',
        target_subtype=['LUSC', 'LUAD'],
        label_column = 'TMB (nonsynonymous)', # or 'Mutation Count'
        tmb_high_ratio = 0.2,
        balance = [1,0,1]
        ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
        """
        # 무조껀 balance[0]은 TMB, balance[1]은 Subtype classification만 하도록  
        if balance[0] and label_column not in ['TMB (nonsynonymous)', 'Mutation Count']:
            raise ValueError('--label_column must be either "TMB (nonsynonymous)" or "Mutation Count" when --loss_balance[0] is True')

        self.slide_data = pd.read_csv(csv_path)[['case_id', 'slide_id', 'label', 'Subtype', 'Mutation Count', 'TMB (nonsynonymous)']]
        self.slide_data = self.filter_df(self.slide_data, filter_dict)
        self.slide_data = self.slide_data[self.slide_data['Subtype'].isin(target_subtype)]

        if label_column in ['TMB (nonsynonymous)', 'Mutation Count']:
            # 각 subtype별로 threshold 몇으로 해야 되는지
            # regression으로 하더라도 split 나눌때, test시에는 threshold 필요함
            self.tmb_threshold = top_n_percent_values(self.slide_data, target_subtype, label_column, tmb_high_ratio)
            # 각 target_subtype에 대해 label_col 찾아서 label_col 값을 0,1 label로 바꿈. tmb_low: 0, tmb_high: 1 
            for subtype, thr in zip(target_subtype, self.tmb_threshold):
                mask = self.slide_data['Subtype'] == subtype
                self.slide_data.loc[mask, label_column] = self.slide_data.loc[mask, label_column].apply(lambda x: label_dict['TMB_high'] if x >= thr else label_dict['TMB_low'])
            self.slide_data[label_column] = self.slide_data[label_column].astype(int) 
        elif label_column in ['Subtype']:
            # 각 sbutype을 string에서 0,1로 바꿈
            self.slide_data[label_column] = self.slide_data[label_column].map(label_dict_subtype)
        # label_column 변환 잘 됐는지 확인
        if self.slide_data[label_column].isna().any():
            raise ValueError("NaN values detected in the 'label_column' column. Please check your label_dict.")

        # print(self.slide_data[label_col])
        self.label_dict = label_dict if balance[0] else label_dict_subtype
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.label_column = label_column
        
        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.slide_data)

        # unique한 patients의 case_id와 그 label 만듦. self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}
        self.patient_data_prep(patient_voting)		
        # unique한 patients의 class별 idx와 각 slide의 class별 idx 구함. self.patient_cls_ids, self.slide_cls_ids
        self.cls_ids_prep()							

        if print_info:
            self.summarize()						# print

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df
    
    def patient_data_prep(self, patient_voting='max'):
        '''
        patient 별로 하나의 label을 달아 dictionary로 만듦.
        label_column의 label을 따름
        '''
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data[self.label_column][locations].values
            if patient_voting == 'max':
                label = label.max() # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, self.label_column:np.array(patient_labels)}

    def cls_ids_prep(self):
        '''
        patient_cls_ids[i]에 label_column == i인 patient index들 저장
        slide_cls_ids[i]에 label_column == i인 slide index들 저장
        '''
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]		
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data[self.label_column] == i)[0]

        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_column] == i)[0]
        
    def summarize(self):
        print("label column: {}".format(self.label_column))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data[self.label_column].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))
    
    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k, 
                    'val_num' : val_num,        # class 별로 몇 개인지 list 형태로
                    'test_num': test_num,
                    'label_frac': label_frac,
                    'seed': self.seed,
                    'custom_test_ids': custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            # self.slide_cls_id에는 []
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 

            # case_id로 나누었던 idx를 다시 slide_idx로 변경 
            for split in range(len(ids)): 
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def test_split_gen(self, return_descriptor=False):
        '''
        train, val, test 각 class 별로 몇 개인지 excel file 저장
        '''
        if return_descriptor:
            # dictionary 순서가 없어서 순서대로 label_dict.keys() list 만들어줌
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                            columns= columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]
        
        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(self.train_ids) > 0
        assert len(self.val_ids) > 0
        assert len(self.test_ids) > 0
        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
            else:
                train_split = None
            
            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
            else:
                val_split = None
            
            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)

        return train_data, val_data, test_data

    def getlabel(self, ids):
        return self.slide_data[self.label_column][ids]
    
    # # 안씀
    # def get_list(self, ids):
    #     return self.slide_data['slide_id'][ids]
        


############################################################################################
############################################################################################
############################################################################################
class Multi_Task_Dataset(Dataset):
    '''
    230926
        output으로 subtype이랑 tmb label이랑 같이 출력
        dataset에서 split까지 나누도록
        regression도 가능하도록
        LUSC, LUAD 중 하나만도 가능하도록

    Args:
        one_subtype: Setting 'LUSC' or 'LUAD' will create a dataset with a single subtype.
        label_column: The label column to use. [label, TMB (nonsynonymous)]
                      Default: label (TMB_low, TMB_high) -> TMB (nonsynonymous)로 변경하여 돌려볼 것
    '''
    def __init__(
            self, 
            csv_path, 
            split_path, 
            data_dir,
            shuffle,        # 여기서 안하고 loader하면 됨
            seed, 
            print_info,
            label_dict, 
            label_dict2, 
            use_h5=True, 
            target_subtype=['LUSC', 'LUAD'],
            label_column='TMB (nonsynonymous)',
            tmb_threshold = [0.5,0.5],
            regression = False,
            split_key = 'train',
            balance = [1,0,1]
    ):
        
        self.slide_data = pd.read_csv(csv_path)[['case_id', 'slide_id', 'label', 'Subtype', 'Mutation Count', 'TMB (nonsynonymous)']]
        self.slide_data.reset_index(drop=True, inplace=True)
        # target_subtype 리스트에 있는 데이터만 필터링
        self.slide_data = self.slide_data[self.slide_data['Subtype'].isin(target_subtype)]
        self.split_path = split_path        # target_subtype split 데이터만
        self.data_dir = data_dir
        self.label_dict = label_dict        # tmb label
        self.label_dict2 = label_dict2      # subtype label
        self.use_h5 = use_h5
        self.num_classes = len(label_dict)
        self.seed = seed
        self.label_column = label_column
        self.regression = regression
        self.split_key = split_key
        self.balance = balance

        # self.slide_data 중에서 split_key(train, val, test 중 하나)만 고름
        all_splits = pd.read_csv(split_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        mask = (self.slide_data['case_id']+'/'+self.slide_data['slide_id']).isin(split.tolist())      # clam  
        self.slide_data = self.slide_data[mask].reset_index(drop=True)
        self.length = len(self.slide_data)
        # exit()

        # regression이 아니라면 slide_data[label_column]을 0,1로 변경
        if not regression and tmb_threshold is not None:
            self.tmb_threshold = tmb_threshold
            # 각 target_subtype에 대해 label_col 찾아서 label_col 값을 0,1 label로 바꿈. tmb_low: 0, tmb_high: 1 
            for subtype, thr in zip(target_subtype, self.tmb_threshold):
                mask = self.slide_data['Subtype'] == subtype
                self.slide_data.loc[mask, label_column] = self.slide_data.loc[mask, label_column].apply(lambda x: label_dict['TMB_high'] if x >= thr else label_dict['TMB_low'])
            self.slide_data[label_column] = self.slide_data[label_column].astype(int) 
        # Subtype column을 string에서 int로 변경
        self.slide_data['Subtype'] = self.slide_data['Subtype'].map(self.label_dict2)

        # label 변환을 __get_item__에서 할거라 사용 불가
        self.cls_ids_prep()
        if print_info:
            self.summarize()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data[self.label_column][idx]
        label2 = self.slide_data['Subtype'][idx]
        if type(self.data_dir) == dict:
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:       # 이게 없을 수가 있나??
                # full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))	# 이게 원래 CLAM 인듯?
                full_path = os.path.join(data_dir, 'pt_files','{}/{}.pt'.format(case_id, slide_id)).replace('.svs', '') # 종성님 CLAM - 이 방식으로 해야됨
                features = torch.load(full_path)
                # print(label, label2)
                return features, label, label2
            
            else:
                return slide_id, label

        else:	# patch coordinates까지 포함되어 있음. coordinate 정보까지 이용하면 도움될 것 같은데 사용 안하고 있음
            full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return features, label, coords

    def cls_ids_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_subtype_cls_ids = [[] for i in range(len(self.label_dict2))]
        for i in range(len(self.label_dict2)):
            self.slide_subtype_cls_ids[i] = np.where(self.slide_data['Subtype'] == i)[0]
        # If not regression, store ids corresponding each tmb class at the slide level
        if not self.regression and self.label_column in ['TMB (nonsynonymous)', 'Mutation Count']:
            self.slide_cls_ids = [[] for i in range(len(self.label_dict))]
            for i in range(len(self.label_dict)):
                # self.slide_cls_ids[i] = np.where(self.slide_data[self.label_column] == list(self.label_dict.keys())[i])[0]
                self.slide_cls_ids[i] = np.where(self.slide_data[self.label_column] == i)[0]
        else:
            # weighted loss 구할 때 self.slide_cls_ids 필요
            # regression이더라도 subtype별로 weighted loss하고자 하면 필요할까봐 일단 변수 만들어 놓음
            self.slide_cls_ids = self.slide_subtype_cls_ids
            # print(self.slide_cls_ids)
            

    def summarize(self):
        if self.balance[1]:
            for i in range(len(self.label_dict2)):
                print(f'{self.split_key} Slide-LVL; Number of samples registered in subtype class {i}: {self.slide_subtype_cls_ids[i].shape[0]}')

        if self.balance[0] and not self.regression and self.label_column in ['TMB (nonsynonymous)', 'Mutation Count']:
            for i in range(len(self.label_dict)): 
                print(f'{self.split_key} Slide-LVL; Number of samples registered in tmb class {i}: {self.slide_cls_ids[i].shape[0]}')

    def getlabel(self, ids):
        '''
        이렇게 하면 label_column에서 가져오는데 regression에서는 int로 안바꿨으므로 문제 생김
        regression에서는 굳이 이 함수를 부르게 되면 tmb에 대한 label이 아니라 
        '''
        return self.slide_data[self.label_column][ids]
        
    

##############################################  CLIP  ####################################################
##############################################  CLIP  ####################################################
##############################################  CLIP  ####################################################
##############################################  CLIP  ####################################################
##############################################  CLIP  ####################################################
##############################################  CLIP  ####################################################


class Split_Gene_Clip_Dataset(Dataset):
    def __init__(self,
        wsi_csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv',
        genomics_csv_path = '/shared/js.yun/data/CLAM_data/genomics_data/TCGA-lung-LUAD+LUSC-selected_2847_zscore.csv',        
        seed = 1,
        num_splits=10,
        val_frac=0.1, 
        test_frac=0.2
        ):
        self.seed = seed
        self.num_splits = num_splits
        self.val_frac=val_frac
        self.test_frac=test_frac
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.slide_data = pd.read_csv(wsi_csv_path)[['case_id', 'slide_id', 'label', 'Subtype', 'Mutation Count', 'TMB (nonsynonymous)']]
        self.genomics_data = pd.read_csv(genomics_csv_path)
        
        self.num_data = self.genomics_data.shape[1]                         # 907명
        self.num_val = np.round(self.num_data * val_frac).astype(int)       # 91
        self.num_test = np.round(self.num_data * test_frac).astype(int)     # 181
        self.num_train = self.num_data-self.num_val-self.num_test           # 635
        
        # print(f'number of train set: {self.num_train}')                 # 635
        # print(f'number of validation set: {self.num_val}')              # 91
        # print(f'number of test set: {self.num_test}')                   # 181
        # df로 저장
        columns = ['train', 'val', 'test']
        num_each_splits = [self.num_train, self.num_val, self.num_test]
        count_dataset = pd.DataFrame([num_each_splits], columns=columns)
        # display(count_dataset)

    def create_splits_index(self):        
        for i in range(self.num_splits):
            all_indices = np.arange(self.num_data).astype(int)
            val_index = np.random.choice(all_indices, self.num_val, replace = False) 
            remaining_ids = np.setdiff1d(all_indices, val_index)
            test_index = np.random.choice(remaining_ids, self.num_test, replace = False) 
            train_index = np.setdiff1d(remaining_ids, test_index)

            if self.val_frac > 0:
                assert len(val_index) > 0
            if self.test_frac > 0:
                assert len(test_index) > 0
            assert len(train_index) > 0
            assert len(np.intersect1d(train_index, test_index)) == 0
            assert len(np.intersect1d(train_index, val_index)) == 0
            assert len(np.intersect1d(val_index, test_index)) == 0

            yield train_index, val_index, test_index

    def create_split_file_name(self, train_index, val_index, test_index):
        train_data = self.genomics_data.columns[train_index]
        val_data = self.genomics_data.columns[val_index]
        test_data = self.genomics_data.columns[test_index]

        train_data = pd.DataFrame(train_data, columns=['case_id'])
        val_data = pd.DataFrame(val_data, columns=['case_id'])
        test_data = pd.DataFrame(test_data, columns=['case_id'])

        return train_data, val_data, test_data



class Clip_dataset(Dataset):
    def __init__(self,
        split_csv_path = '/shared/js.yun/data/CLAM_data/clip_data/TCGA-lung-splits_5-frac_1_0_0-seed0/splits_0.csv',
        genomics_csv_path = '/shared/js.yun/data/CLAM_data/genomics_data/TCGA-lung-LUAD+LUSC-selected_2847_zscore.csv',        
        wsi_csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv',
        wsi_feature_dir = '/shared/j.jang/pathai/data/TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/',
        phase = 'train',
        seed = 1,
        ):
        self.wsi_feature_dir = wsi_feature_dir
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # genomics dataset
        self.selected_columns = list(pd.read_csv(split_csv_path)[phase].dropna())                        # training set에 있는 환자 set
        genomics_data = pd.read_csv(genomics_csv_path)
        self.genomics_data = genomics_data.loc[:, genomics_data.columns.isin(self.selected_columns)]    # self.selected_columns에 있는 환자의 genomics만 가져온 df
        self.length = self.genomics_data.shape[1]

        # WSI dataset
        slide_data = pd.read_csv(wsi_csv_path)[['case_id', 'slide_id']]
        slide_data['patient'] = slide_data['slide_id'].str.split('-').apply(lambda x: '-'.join(x[:3] + [x[3][:2]]))
        self.slide_data = slide_data[slide_data['patient'].isin(self.selected_columns)]                 # csv 파일 중에서 self.selected_columns에 있는 환자의 csv만 가져온 df

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 불러올 환자 선택
        selected_column = list(self.selected_columns)[idx]

        # genomics data 불러옴
        genomics = self.genomics_data[selected_column].to_numpy()

        # image data 불러옴
        indices_wsi = self.slide_data.index[self.slide_data['patient'] == selected_column].tolist()
        if len(indices_wsi) > 1:
            index = random.choice(indices_wsi)
        else:
            index = indices_wsi[0]
        case_id = self.slide_data['case_id'][index]
        slide_id = self.slide_data['slide_id'][index]
        full_path = os.path.join(self.wsi_feature_dir, 'pt_files','{}/{}.pt'.format(case_id, slide_id)).replace('.svs', '') # 종성님 CLAM - 이 방식으로 해야됨
        features = torch.load(full_path)

        return features, genomics
    

class Genomics_TMB_dataset(Dataset):
    def __init__(self,
        split_csv_path = '/shared/js.yun/data/CLAM_data/clip_data/TCGA-lung-splits_5-frac_1_0_0-seed0/splits_0.csv',
        genomics_csv_path = '/shared/js.yun/data/CLAM_data/genomics_data/TCGA-lung-LUAD+LUSC-selected_2847_zscore.csv',        
        wsi_csv_path = '/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv',
        wsi_feature_dir = '/shared/j.jang/pathai/data/TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/',
        phase = 'train',
        seed = 1,
        ):
        self.wsi_feature_dir = wsi_feature_dir
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # genomics dataset
        self.selected_columns = set(pd.read_csv(split_csv_path)[phase].dropna())                      # training set에 있는 환자 set
        genomics_data = pd.read_csv(genomics_csv_path)
        self.genomics_data = genomics_data.loc[:, genomics_data.columns.isin(self.selected_columns)]
        self.length = self.genomics_data.shape[1]

        # TMB dataset
        slide_data = pd.read_csv(wsi_csv_path)[['case_id', 'slide_id', 'Mutation Count']]
        slide_data['patient'] = slide_data['slide_id'].str.split('-').apply(lambda x: '-'.join(x[:3] + [x[3][:2]]))
        self.slide_data = slide_data[slide_data['patient'].isin(self.selected_columns)]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 불러올 환자 선택
        selected_column = list(self.selected_columns)[idx]
        
        # genomics data 불러옴
        genomics = self.genomics_data[selected_column].to_numpy()

        # mutation count 값 불러옴
        indices = self.slide_data.index[self.slide_data['patient'] == selected_column].tolist()
        if len(indices) > 1:
            index = random.choice(indices)
        else:
            index = indices[0]
        mutation_count = self.slide_data['Mutation Count'][index]
        
        return genomics, mutation_count