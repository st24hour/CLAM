import torch
import torch.nn as nn
from models.model_clip import CLIP
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets.dataset import Clip_dataset
import math
from torch.optim.lr_scheduler import _LRScheduler
from utils.utils import setup_logger
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloader(args, dataset, is_train = True):
    # assuming DP
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = args.per_gpu_train_batch_size * args.num_gpu       
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.per_gpu_eval_batch_size * args.num_gpu

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.num_workers)

    return dataloader

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter, base_lr, final_lr, total_iters, last_epoch=-1):
        self.warmup_iter = warmup_iter
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iter:
            return [base_lr * (self.last_epoch / self.warmup_iter) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cos_out = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_iter) / (self.total_iters - self.warmup_iter)))
            return [cos_out]


def train_wrapper(split_csv_path, fold, args):
    save_dir_fold = os.path.join(args.save_dir, str(fold))
    os.makedirs(save_dir_fold, exist_ok=True)

    # logger
    logger = setup_logger("CLIP", save_dir_fold, 0, filename = "training_logs.txt")         # 나중에 필요하면 rank 넣어줘야됨

    # dataset
    train_dataset = Clip_dataset(split_csv_path, args.genomics_csv_path, args.wsi_csv_path, args.wsi_feature_dir, 'train', args.seed)
    val_dataset = Clip_dataset(split_csv_path, args.genomics_csv_path, args.wsi_csv_path, args.wsi_feature_dir, 'val', args.seed)
    test_dataset = Clip_dataset(split_csv_path, args.genomics_csv_path, args.wsi_csv_path, args.wsi_feature_dir, 'test', args.seed)
    logger.info("Training on {} samples".format(len(train_dataset)))
    logger.info("Validating on {} samples".format(len(val_dataset)))
    logger.info("Testing on {} samples".format(len(test_dataset)))

    # dataloader
    train_loader = get_dataloader(args, train_dataset, is_train=True)
    val_loader = get_dataloader(args, val_dataset, is_train=False)
    test_loader = get_dataloader(args, test_dataset, is_train=False)

    # model
    model = CLIP()  # input 넣어줘야됨

    # loss
    loss_img = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wd) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = WarmupCosineLR(optimizer, warmup_iter=args.warmup_iter, base_lr=args.lr, final_lr=1e-06, total_iters=len(train_loader)*args.epochs) # paper, github에 final_lr 정보 없음

    # training
    for epoch in range(args.epochs):
        train(train_loader, model, loss_img, loss_text, optimizer, scheduler)

        # 중간에 평가 어떻게???
        # batch size 개수만큼씩 넣고 image랑 genomics랑 pair 맞는지 확인. batch size에 따라 달라질 수 있긴함

    return



def train(dataloader, model, loss_img, loss_text, optimizer, scheduler):
    
    model.train()
    model.zero_grad()

    epoch_loss = 0.
    for iter, batch in enumerate(dataloader):
        images, text = batch
        images = images.to(device)
        text = text.to(device)

        logits_per_image, logits_per_text = model(images, text)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_text(logits_per_text,ground_truth))/2
        total_loss.backward()

        epoch_loss += total_loss.item()

        # optimizer step
        optimizer.step() 
        scheduler.step()
        model.zero_grad()

        # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # logging
        
    return 



def test(dataloader, model, loss_img, loss_text):
    model.test()
    

    return 