from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data. 여기서는 이미지가 h5로 저장해놓은 폴더
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		# print(np.shape(img))

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




class Fast_Dataset_v2(Dataset):
    '''
    이미지 불러와서 to_tensro 안하고 torch.from_numpy(img.transpose((2, 0, 1))).contiguous()만 하여 return 
    ByteTensor로 GPU까지 전달 (float보다 전달도 더 빠름)
    float32로 바꾸고 255로 나누는 과정 없음
    transform 없음 -> GPU에서 수행

    230831:
        read_h5 = False로 하더라도 buffer에서 바로 tensor로 바꾸도록 수정하였음
    230904:
        tensor 메모리로 읽어오도록 openslide_to_tensor.cpp 만들었는데 python에서 하는거보다 비슷하거나 느림 -> v2로 쓰는게 나음
    '''
    def __init__(self, csv_file, wsis, is_training, h5_file_paths=None, read_h5=True):
        '''
        Args:
            csv_file:
			  
        '''
        self.slide_data = pd.read_csv(csv_file)[['case_id', 'slide_id', 'label' ]]  # 모든 slide
        self.slide_data.reset_index(drop=True, inplace=True)
        self.wsis = wsis                    # train 데이터만
        self.read_h5 = read_h5

        ### prepare labels
        temp = pd.DataFrame()
        for i in range(len(self.wsis)):
            for k in range(len(self.slide_data)):
                if self.wsis[i].split('/')[-1] == self.slide_data['slide_id'][k]:   # split_csv는 train만, csv_path에는 모든 slide
                    temp=temp.append(self.slide_data.loc[k])
                    break
        self.slide_data = temp
        self.slide_data.reset_index(inplace=True)
        self.slide_data = self.slide_data.drop(['index'], axis=1)
        # print(len(self.slide_data))     # TGCA-lung: train set 842장 (= self.length)

        self.h5_file_paths = h5_file_paths
        self.dset = np.array([], dtype=np.int64).reshape(0, 2)  # empty array
        self.labels = []  # accumulated number of batches
        self.slide_idx = []
        self.slide_labels = []
        if read_h5:
            self.patch_num_list = [0]   # h5에서 image read 할때만 사용됨
        for idx, h5_file_path in enumerate(self.h5_file_paths):         # patch4096_dir (파일이 slide 개수만큼 있고 좌표 뽑아놓은 파일)
            with h5py.File(h5_file_path, "r") as f:
                if not read_h5:                                         # read from WSI
                    _num_patch = f['coords'].shape[0]
                    # print(_num_patch)
                    self.dset = np.vstack([self.dset, np.array(f['coords'])])
                    self.patch_level = f['coords'].attrs['patch_level']
                    # self.patch_size = f['coords'].attrs['patch_size']
                    self.length = self.dset.shape[0]
                else:                                                   # read from saved h5 file
                    _num_patch = f['imgs'].shape[0]
                    self.patch_num_list.append(_num_patch+self.patch_num_list[-1])
                self.slide_idx.extend([idx]*_num_patch)
                self.length = len(self.labels)
        # self.length = self.dset.shape[0]
        # print(self.length)              # 148704
        # print(self.dset.shape)      # (148704, 2)
        # print(self.labels)
        # print(len(self.labels))         # 148704
        # print(self.slide_idx)
        # print(len(self.slide_idx))      # 148704

        self.cls_ids_prep()

        self.is_training = is_training     

    def __len__(self):
        return self.length

    def __getitem__(self, idx):    
        # 여기서 idx로 여는게 아니라 patch idx가 속하는 slide를 찾아서 열어야됨 
        if self.read_h5:
            with h5py.File(self.h5_file_paths[self.slide_idx[idx]], "r") as f:
                wsi_id=self.slide_idx[idx]
                cur_patch_idx = idx-self.patch_num_list[wsi_id]
                img = f['imgs'][cur_patch_idx]
                img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()

        slide_idx = self.slide_idx[idx]     # 해당 patch가 몇 번째 slide인지 idx로 출력

        # worker_info = torch.utils.data.get_worker_info()
        return img, slide_idx


