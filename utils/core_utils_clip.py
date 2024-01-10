import os
import torch
import torch.nn as nn
import models.model_clip as clip_models
from torch.optim import AdamW, SGD
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets.dataset import Clip_dataset
import math
from torch.optim.lr_scheduler import _LRScheduler
from utils.utils import setup_logger, calculate_accuracy, calculate_paired_accuracy, combine_dicts
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
device = "cuda" if torch.cuda.is_available() else "cpu"


def add_weight_decay(model, wd):
    param_wo_wd, param_with_wd = [],[]
    for name, param in model.named_parameters():
        if len(param.size())<=1:
            param_wo_wd.append(param)
        else:
            param_with_wd.append(param)
    return [
        {'params': param_wo_wd, 'weight_decay': 0.},
        {'params': param_with_wd}]

def collate_wsi(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    text = torch.cat([torch.from_numpy(item[1]) for item in batch], dim = 0)
    # img = torch.cat([item[0] for item in batch], dim = 0)
    # text = torch.LongTensor([item[1] for item in batch])
    return [img, text]

def get_dataloader(args, dataset, is_train = True):
    if is_train:
        sampler = RandomSampler(dataset)
        # batch_size = args.per_gpu_train_batch_size * args.num_gpu       
    else:
        sampler = SequentialSampler(dataset)
        # batch_size = args.per_gpu_eval_batch_size * args.num_gpu

    batch_size = 1
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.num_workers, collate_fn = None if args.use_batch else collate_wsi)

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

class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass  # 아무 작업도 수행하지 않습니다.

def save_checkpoint(args, model, epoch, path):
    torch.save({
        'args': args,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }, path)

def train_wrapper(args, split_csv_path, fold):
    save_dir_fold = os.path.join(args.save_dir, str(fold))
    os.makedirs(save_dir_fold, exist_ok=True)
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(save_dir_fold, flush_secs=15)
    else:
        writer = None

    # logger
    logger = setup_logger("CLIP", save_dir_fold, 0, filename = "training_logs.txt")         # 나중에 필요하면 rank 넣어줘야됨
    logger.info(args)

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
    call_model_clip = getattr(clip_models, args.model, None)
    if callable(call_model_clip):
        model_clip = call_model_clip(image_encoder_name = args.image_encoder,
                                     text_encoder_name = args.genomics_encoder,
                                     size_arg='custom2_big', 
                                     output_dim=args.output_dim,
                                     input_dim_genom = 2847,
                                     hidden_dim_genom = args.hidden_dim_genom,
                                     num_layers_genom = args.num_layers_genom, 
                                     activation_func_genom = args.activation_func_genom, 
                                     norm_genom = args.norm_genom)
    else:
        print(f"{args.model} does not exist or is not callable.")
    model_similarity = clip_models.CLIP_similaritys()
    model = nn.ModuleList([model_clip, model_similarity])
    model.to(device)

    # loss
    loss_img = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    
    # optimizer
    parameters = add_weight_decay(model, args.wd)
    if args.optimizer in ['AdamW', 'adamw']:
        optimizer = AdamW(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wd) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    elif args.optimizer in ['SGD', 'sgd']:
        optimizer = SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.scheduler in ['cos', 'cosine']:
        final_lr = args.lr * 0.01      # 1e-06 이었다가 수정함
        scheduler = WarmupCosineLR(optimizer, warmup_iter=args.warmup_iter, base_lr=args.lr, final_lr=final_lr, total_iters=len(train_loader)*(args.epochs)) # paper, github에 final_lr 정보 없음
    elif args.scheduler in ['None']:
        scheduler = DummyScheduler(optimizer)
    else:
        raise NotImplementedError('Only cosine scheduler is supported')

    # training
    best_acc = 0
    for epoch in range(args.epochs):
        best_model_state = model.state_dict()
        best_val_dict = {}
        # train(args, logger, writer, epoch, train_loader, model, loss_img, loss_text, optimizer, scheduler)
        val_dict = test(args, logger, writer, epoch, val_loader, model, loss_img, loss_text, best_acc)
        if val_dict['acc_pair'] > best_acc:
            best_val_dict = {
                "acc_img": val_dict['acc_img'],
                "acc_txt": val_dict['acc_txt'],
                "acc_pair": val_dict['acc_pair'],
            }
            path = os.path.join(save_dir_fold, 'model_best.pth')
            best_model_state = model.state_dict()
            save_checkpoint(args, model, epoch, path)

    # test final
    logger.info(f'Test Final Model')
    test_dict = test(args, logger, writer, args.epochs, test_loader, model, loss_img, loss_text, best_acc)
    save_checkpoint(args, model, args.epochs, os.path.join(save_dir_fold, 'model_final.pth'))
    
    # test best
    # checkpoint = torch.load(path)
    # model_state_dict = checkpoint['model_state_dict']
    # model.load_state_dict(model_state_dict)
    model.load_state_dict(best_model_state)
    logger.info(f'Test Best Model')
    best_test_dict = test(args, logger, None, args.epochs, test_loader, model, loss_img, loss_text, best_acc)
    if writer:
        writer.add_scalar(f'test_best/acc_img', best_test_dict['acc_img'], epoch)
        writer.add_scalar(f'test_best/acc_txt', best_test_dict['acc_txt'], epoch)
        writer.add_scalar(f'test_best/acc_pair', best_test_dict['acc_pair'], epoch)

    # 중간에 평가 어떻게???
    # batch size 개수만큼씩 넣고 image랑 genomics랑 pair 맞는지 확인. batch size에 따라 달라질 수 있긴함

    return combine_dicts(val_dict, test_dict, best_val_dict, best_test_dict)    # final_val, final_test, best_val, best_test



def train(args, logger, writer, epoch, dataloader, model, loss_img, loss_text, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()
    model_clip, model_similarity = model

    epoch_loss = 0.
    epoch_acc_img, epoch_acc_txt, epoch_acc_pair = 0,0,0
    image_list, text_list = [], []
    scaler = GradScaler()

    i=0
    for iter, batch in enumerate(dataloader):
        if iter == 200:
            break
        if iter !=0 and iter % args.batch_size == 0:
            with autocast():
                image_features = torch.stack(image_list)
                text_features = torch.stack(text_list)
                logits_per_image, logits_per_text = model_similarity(image_features, text_features)
            
            # loss
            ground_truth = torch.arange(args.batch_size,dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_text(logits_per_text,ground_truth))/(2*args.batch_size)

            scaler.scale(total_loss).backward()
            # total_loss.backward()
            epoch_loss += total_loss.item()
            
            # accuracy - The value depends on the batch size. just consider it as a proxy for good training
            accuracy_image = calculate_accuracy(logits_per_image.detach(), ground_truth)
            accuracy_text = calculate_accuracy(logits_per_text.detach(), ground_truth)
            accuracy_pair = calculate_paired_accuracy(logits_per_image.detach(), logits_per_text.detach(), ground_truth)
            epoch_acc_img += accuracy_image
            epoch_acc_txt += accuracy_text
            epoch_acc_pair += accuracy_pair

            logger.info(f'Epoch {epoch}/{args.epochs} iter {i} loss {total_loss.item():.4f} '
                        f'acc_pair {accuracy_pair:.4f} acc_img {accuracy_image:.4f}, acc_txt {accuracy_text:.4f}')

            # optimizer step
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
            model_similarity.logit_scale.data = torch.clamp(model_similarity.logit_scale.data, 0, 4.6052)

            image_list[:] = []
            text_list[:] = []
            i+=1

        images, text = batch
        images = images.to(device)
        text = text.to(device).float()

        # feature extraction
        with autocast():
            image_feature, text_feature = model_clip(images, text)
            
            image_list.append(image_feature.squeeze())
            text_list.append(text_feature.squeeze())

    # logging
    logger.info(f'Train epoch {epoch}/{args.epochs} avg loss {epoch_loss/i:.4f} '
                f'acc_pair {epoch_acc_pair/i:.4f} acc_img {epoch_acc_img/i:.4f} acc_txt {epoch_acc_txt/i:.4f}')
    if writer:
        writer.add_scalar(f'train/loss', epoch_loss/i, epoch)
        writer.add_scalar(f'train/acc_img', epoch_acc_img/i, epoch)
        writer.add_scalar(f'train/acc_txt', epoch_acc_txt/i, epoch)
        writer.add_scalar(f'train/acc_pair', epoch_acc_pair/i, epoch)

    return


def test(args, logger, writer, epoch, dataloader, model, loss_img, loss_text, best_acc):
    model.eval()
    model_clip, model_similarity = model

    epoch_loss = 0.
    epoch_acc_img, epoch_acc_txt, epoch_acc_pair = 0,0,0
    image_list, text_list = [], []

    i=0
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):

            if iter !=0 and iter % args.batch_size == 0:
                with autocast():
                    image_features = torch.stack(image_list)
                    text_features = torch.stack(text_list)
                    logits_per_image, logits_per_text = model_similarity(image_features, text_features)

                # loss
                ground_truth = torch.arange(args.batch_size,dtype=torch.long,device=device)
                total_loss = (loss_img(logits_per_image,ground_truth) + loss_text(logits_per_text,ground_truth))/(2*args.batch_size)
                epoch_loss += total_loss.item()
                
                # accuracy - The value depends on the batch size. just consider it as a proxy for good training
                accuracy_image = calculate_accuracy(logits_per_image.detach(), ground_truth)
                accuracy_text = calculate_accuracy(logits_per_text.detach(), ground_truth)
                accuracy_pair = calculate_paired_accuracy(logits_per_image.detach(), logits_per_text.detach(), ground_truth)
                epoch_acc_img += accuracy_image
                epoch_acc_txt += accuracy_text
                epoch_acc_pair += accuracy_pair

                logger.info(f'{"Test" if epoch==args.epochs else "Val"} epoch {epoch}/{args.epochs} iter {i} loss {total_loss.item():.4f} '
                            f'acc_pair {accuracy_pair:.4f} acc_img {accuracy_image:.4f}, acc_txt {accuracy_text:.4f}')

                image_list[:] = []
                text_list[:] = []
                i+=1

            images, text = batch
            if epoch==args.epochs and iter==0:
                print(images[0][0][0:10])
            images = images.to(device)
            text = text.to(device).float()

            # feature extraction
            image_feature, text_feature = model_clip(images, text)
            image_list.append(image_feature.squeeze())
            text_list.append(text_feature.squeeze())

    # logging
    epoch_acc_img = epoch_acc_img/i
    epoch_acc_txt = epoch_acc_txt/i
    epoch_acc_pair = epoch_acc_pair/i
    logger.info(f'{"Test" if epoch==args.epochs else "Val"} Epoch {epoch}/{args.epochs} '
                f'avg loss {epoch_loss/i:.4f} '
                f'acc_pair {epoch_acc_pair:.4f} acc_img {epoch_acc_img:.4f} acc_txt {epoch_acc_txt:.4f}')
    if writer:
        writer.add_scalar(f'val/loss' if not epoch==args.epochs else "test/loss", epoch_loss/i, epoch)
        writer.add_scalar(f'val/acc_img' if not epoch==args.epochs else "test/acc_img", epoch_acc_img, epoch)
        writer.add_scalar(f'val/acc_txt' if not epoch==args.epochs else "test/acc_txt", epoch_acc_txt, epoch)
        writer.add_scalar(f'val/acc_pair' if not epoch==args.epochs else "test/acc_pair", epoch_acc_pair, epoch)

    output_dict = {
        "acc_img": epoch_acc_img,
        "acc_txt": epoch_acc_txt,
        "acc_pair": epoch_acc_pair,
    }
    return output_dict