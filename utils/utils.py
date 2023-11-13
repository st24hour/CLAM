import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_multi_task(batch):
    if len(batch[0]) == 3:
        img = torch.cat([item[0] for item in batch], dim = 0)
        label = torch.LongTensor([item[1] for item in batch])
        label2 = torch.LongTensor([item[2] for item in batch])
        return [img, label, label2]
    else:
        img = torch.cat([item[0] for item in batch], dim = 0)
        label = torch.LongTensor([item[1] for item in batch])
        return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), 
						collate_fn = collate_multi_task, **kwargs)	# multi_task
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_multi_task, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_multi_task, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'adamw':
		optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)	
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)        # samples는 patient수(한 환자 slide가 여러개 있을 수 있어서 slide 수랑 다름)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):   # class 수 만큼 for문 돎
			possible_indices = np.intersect1d(cls_ids[c], indices) # all indices of this class, cls_ids[c]에는 c번 class의 index 들어있음
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))           
    # print(dataset.slide_cls_ids[0])
    # print(dataset.slide_cls_ids[1])
    # exit()
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  
    return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			if m.bias is not None:
				m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


# def top_n_percent_values(lst, n):
#     if not 0 <= n <= 100:
#         raise ValueError("n should be between 0 and 100")

#     lst.sort(reverse=True)  # 리스트를 내림차순으로 정렬합니다.
#     k = int(len(lst) * (n / 100))  # 리스트 길이의 n%에 해당하는 인덱스를 계산합니다.
    
#     # 상위 n% 값들을 반환합니다.
#     top_n_percent = lst[:k]
#     return top_n_percent

def print_4f(float_input):
    return f'{float_input:.4f}'


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    import os 
    import sys
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    streamHandler = logging.StreamHandler(stream=sys.stdout)
    streamHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    if save_dir:
        fileHandler = logging.FileHandler(os.path.join(save_dir, filename))
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger