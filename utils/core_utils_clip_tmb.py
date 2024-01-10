import os
import torch
import torch.nn as nn
import models.model_clip as clip_models
from torch.optim import AdamW, SGD
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets.dataset import Clip_dataset
from utils.core_utils_clip import add_weight_decay, get_dataloader, save_checkpoint, WarmupCosineLR, DummyScheduler
from utils.utils import setup_logger, calculate_accuracy, calculate_paired_accuracy, combine_dicts
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
device = "cuda" if torch.cuda.is_available() else "cpu"


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
        train(args, logger, writer, epoch, train_loader, model, loss_img, loss_text, optimizer, scheduler)
        val_dict = test(args, logger, writer, epoch, val_loader, model, loss_img, loss_text, best_acc)
        if val_dict['acc_pair'] > best_acc:
            best_val_dict = {
                "acc_img": val_dict['acc_img'],
                "acc_txt": val_dict['acc_txt'],
                "acc_pair": val_dict['acc_pair'],
            }
            path = os.path.join(save_dir_fold, 'model_best.pth')
            save_checkpoint(args, model, epoch, path)

    # test final
    test_dict = test(args, logger, writer, args.epochs, test_loader, model, loss_img, loss_text, best_acc)
    save_checkpoint(args, model, args.epochs, os.path.join(save_dir_fold, 'model_final.pth'))
    
    # test best
    checkpoint = torch.load(path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    best_test_dict = test(args, logger, None, args.epochs, test_loader, model, loss_img, loss_text, best_acc)
    if writer:
        writer.add_scalar(f'test_best/acc_img', best_test_dict['acc_img'], epoch)
        writer.add_scalar(f'test_best/acc_txt', best_test_dict['acc_txt'], epoch)
        writer.add_scalar(f'test_best/acc_pair', best_test_dict['acc_pair'], epoch)

    # 중간에 평가 어떻게???
    # batch size 개수만큼씩 넣고 image랑 genomics랑 pair 맞는지 확인. batch size에 따라 달라질 수 있긴함

    return combine_dicts(val_dict, test_dict, best_val_dict, best_test_dict)    # final_val, final_test, best_val, best_test