from math import floor
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="8, 9" ## must declare before importing torch

import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
from models.HIPT_vit import get_vit256                  # HIPT pretrained model로 feature 뽑기 위해
from models.HIPT_vit import VisionTransformer as vits   # HIPT pretrained model로 feature 뽑기 위해
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import torch
import torch.nn as nn
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def compute_w_loader(t_global, file_path, output_path, wsi, model, image_encoder,
                     batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
                     custom_downsample=1, target_patch_size=-1, custom_transform=None, loader_choice ='default', num_worker=8):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size, custom_transforms=custom_transform)
    x, y = dataset[0]
    # kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    kwargs = {'num_workers': num_worker, 'pin_memory': False} if device.type == "cuda" else {}
    if loader_choice == 'default':
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
    elif loader_choice == 'multi_epoch':
        loader = MultiEpochsDataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
    elif loader_choice == 'multi_epoch2':
        loader = MultiEpochsDataLoader2(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
    else:
        raise NotImplementedError

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    
    print(f'before epoch start {time.time()-t_global}')
    for count, (batch, coords) in enumerate(loader):
        print(f'epoch start {time.time()-t_global}')

        with torch.no_grad():	
            # print(f'batch loop start')
            t_batch = time.time()
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            # vit 에서 에러나서 바꿈
            if hasattr(model, 'module'):
                if hasattr(model.module, 'conv1'):
                    if model.module.conv1.weight.dtype == torch.float16:
                        batch=batch.to(torch.float16)
            else:
                if hasattr(model, 'conv1'):
                    if model.conv1.weight.dtype == torch.float16:
                        batch=batch.to(torch.float16)
                
            print(f'before forward {time.time()-t_global}')
            features = model(batch)
            print(f'after forward {time.time()-t_global}')
            # print(f'model forward elapsed time {time.time()-tt}')
            # tt = time.time()
            
            if 'vit' in image_encoder:
                features /= features.norm(dim=-1, keepdim=True)    
                
            features = features.cpu().numpy()
            # print(f'to_cpu_numpy elapsed time {time.time()-tt}')
            # tt = time.time()
            
            asset_dict = {'features': features, 'coords': coords}
            #########################################################################################
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            #########################################################################################
            mode = 'a'
            # print(f'save elapsed time {time.time()-tt}')
            print(f'one batch elapsed time {time.time()-t_batch}')
    print(f'epoch end {time.time()-t_global}')

            
    return output_path

# class MultiEpochsDataLoader(torch.utils.data.DataLoader):
class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MultiEpochsDataLoader2(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/shared/j.jang/pathai/data/TCGA-kidney-patches/', help="Patches directory")
parser.add_argument('--data_slide_dir', type=str, default='/shared/j.jang/pathai/data/TCGA-kidney/', help="DATA_DIRECTORY")
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default='/shared/j.jang/pathai/data/TCGA-kidney-patches/process_list_autogen.csv', help="location of process_list_autogen.csv")
parser.add_argument('--feat_dir', type=str, default='/shared/j.jang/pathai/data/TCGA-kidney-features/', help="FEATURES_DIRECTORY")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--no_auto_skip', default=True, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--image_encoder', type=str, default='resnet50')
parser.add_argument('--model_256_path', type=str, default=None)
parser.add_argument('--loader', type=str, choices=['default', 'multi_epoch', 'multi_epoch2'], default=None)
parser.add_argument('--num_worker', type=int, default=8)
args = parser.parse_args()

###
# args.no_auto_skip = False
# #args.data_h5_dir = '/shared/j.jang/pathai/data/TCGA-lung-patches/'
# #args.data_h5_dir = '/shared/j.jang/pathai/data/TCGA-kidney-patches/'
# args.data_h5_dir = '/shared/j.jang/pathai/data/TCGA-breast-patches/'
# #args.data_slide_dir = '/shared/j.jang/pathai/data/TCGA-lung/'
# #args.data_slide_dir = '/shared/j.jang/pathai/data/TCGA-kidney/'
# args.data_slide_dir = '/shared/j.jang/pathai/data/TCGA-breast/'
# #args.csv_path = '/shared/j.jang/pathai/data/TCGA-lung-patches/process_list_autogen.csv'
# args.csv_path = '/shared/j.jang/pathai/data/TCGA-breast-patches/process_list_autogen.csv'
# #args.feat_dir='/shared/j.jang/pathai/data/TCGA-lung-features/'
# #args.feat_dir='/shared/j.jang/pathai/data/TCGA-breast-features-VITB-16/'
# args.feat_dir='/shared/j.jang/pathai/data/TCGA-breast-features-VITL-14-336px/'
# #args.image_encoder = 'resnet50' #or 'vit_b_16"
# args.image_encoder = 'vit_l_14@336px'
# args.custom_transform = None
###

# JS
# args.data_h5_dir = '/shared/js.yun/data/CLAM_data/TCGA-lung-patches/'
# args.data_slide_dir = '/shared/js.yun/data/CLAM_data/TCGA-lung/'
# args.csv_path = '/shared/js.yun/data/CLAM_data/TCGA-lung-patches/process_list_autogen.csv'
# args.feat_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-features/'
# args.image_encoder = 'resnet50'
args.custom_transform = None

if __name__ == '__main__':
    t_global = time.time()
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    #dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print('loading model checkpoint')
    if args.image_encoder == 'resnet50':
        model = resnet50_baseline(pretrained=True)
    # elif 'vit' in args.image_encoder:
    #     import sys
    #     sys.path.append("/shared/j.jang/pathai/CLIP")
    #     import clip
    #     if args.image_encoder == 'vit_b_16':
    #         model, preprocess = clip.load("ViT-B/16")
    #     elif args.image_encoder == 'vit_l_14':
    #         model, preprocess = clip.load("ViT-L/14")
    #     elif args.image_encoder == 'vit_l_14@336px':
    #         model, preprocess = clip.load("ViT-L/14@336px")
    #     else:
    #         ValueError("not supported image encoder")
    #     model = model.visual
    #     args.custom_transform = preprocess
    elif args.image_encoder == 'HIPT_256':
        assert args.model_256_path
        model = get_vit256(args.model_256_path) # if args.model_256_path else vits.__dict__['vit_small'](patch_size=16, num_classes=0)    
    else:
        ValueError("not supported image encoder")
    model = model.to(device)

    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    # ii = 0
    for bag_candidate_idx in range(total):
        # if ii == 2:
        #     exit()

        tt = time.time()
        if bags_dataset.df.loc[bag_candidate_idx, 'status'] == 'failed_seg':
            continue
        patient_id, wsi_id = bags_dataset[bag_candidate_idx].split('/')
        os.makedirs(os.path.join(args.feat_dir, 'h5_files', patient_id), exist_ok=True)

        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

		#if not args.no_auto_skip and slide_id+'.pt' in dest_files:
        if not args.no_auto_skip and os.path.isfile(os.path.join(args.feat_dir,'pt_files', slide_id + '.pt')):
            print('skipped {}'.format(slide_id))
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(t_global, h5_file_path, output_path, wsi,
											model = model, batch_size = args.batch_size,
											verbose = 1, print_every = 20,
											custom_downsample=args.custom_downsample,
											target_patch_size=args.target_patch_size,
                                            custom_transform = args.custom_transform,
                                            image_encoder = args.image_encoder,
                                            loader_choice = args.loader,
                                            num_worker = args.num_worker)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        # ########################################################################################
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        os.makedirs(os.path.join(args.feat_dir, 'pt_files', patient_id), exist_ok=True)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
        ########################################################################################
        print(f'TOTAL ELAPSED TIME: {time.time()-tt}')
        # ii += 1
        