import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, CLAM_SB_multi, CLAM_MB_multi
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import time
from copy import deepcopy

from utils.utils import print_4f

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=-1)
        probs = (probs.gather(dim=-1, index=target.unsqueeze(-1))).squeeze(-1)  # get the probability of the true class
        focal_weight = (1. - probs).pow(self.gamma)
        loss = -self.alpha * focal_weight * torch.log(probs)
        return loss.mean()
    
class CrossEntropyLabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(CrossEntropyLabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        smoothed_labels = (1.0 - self.epsilon) * F.one_hot(target, n_classes).float() + self.epsilon / n_classes
        log_probs = F.log_softmax(logits, dim=-1)
        loss = (-smoothed_labels * log_probs).sum(dim=-1).mean()
        return loss

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
        
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    # print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    if args.task != 'multi_task':
        if args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
            if device.type == 'cuda':
                loss_fn = loss_fn.cuda()
        elif args.label_smoothing > 0:
            loss_fn = CrossEntropyLabelSmoothingLoss(args.label_smoothing)
        elif args.focal_loss:
            if args.label_smoothing != 0:
                raise ValueError("When using focal_loss, label_smoothing must be set to 0.")
            if args.weighted_sample:
                print("Weighted sample and focal loss are both used")
            loss_fn = FocalLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    elif args.task == 'multi_task':
        if args.loss_balance[0]:
            if args.regression:
                loss_fn = nn.MSELoss()
            elif args.label_smoothing > 0:
                loss_fn = CrossEntropyLabelSmoothingLoss(args.label_smoothing)
            else:
                loss_fn = nn.CrossEntropyLoss()
        if args.loss_balance[1]:
            if args.label_smoothing > 0:
                loss_fn_subtype = CrossEntropyLabelSmoothingLoss(args.label_smoothing)
            else:
                loss_fn_subtype = nn.CrossEntropyLoss()
        else:
            loss_fn_subtype = None
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    model_dict.update({"attn": args.attn})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_multi', 'clam_mb_multi']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)    
        elif args.model_type == 'clam_sb_multi':
            model = CLAM_SB_multi(**model_dict, n_classes_subtype=args.n_classes_subtype, 
                                  instance_loss_fn=instance_loss_fn, balance=args.loss_balance)
        elif args.model_type == 'clam_mb_multi':
            model = CLAM_MB_multi(**model_dict, n_classes_subtype=args.n_classes_subtype, 
                                  instance_loss_fn=instance_loss_fn, balance=args.loss_balance) 
        else:
            raise NotImplementedError
    

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    # model.relocate() # ??? 모듈별로 cuda로 보내야되는 이유가 있는지???
    model.to(device)
    # print_network(model)

    optimizer = get_optim(model, args)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=0.1)
    
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
                                    weighted = args.weighted_sample if not args.regression else False)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)

    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None

    max_auc = 0
    for epoch in range(args.max_epochs):
        scheduler.step(epoch)
        if args.model_type == 'clam_sb_multi' or args.model_type == 'clam_mb_multi':
            train_loop_multi(epoch, model, train_loader, optimizer, args.n_classes, args.n_classes_subtype, \
                             writer, loss_fn, loss_fn_subtype, args.loss_balance)
            stop, max_flag, max_auc = validate_multi(args, cur, epoch, model, val_loader,
                early_stopping, writer, loss_fn, loss_fn_subtype, args.loss_balance, max_auc=max_auc)
            
        elif args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop, max_flag, max_auc = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, max_auc=max_auc)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop, max_flag, max_auc = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, max_auc=max_auc)
        

        # best auc를 달성한 경우 best_model로 저장
        if max_flag:
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(writer_dir, f"max_auc_checkpoint.pt"))

        if stop: 
            break

    # save final model
    torch.save(model.state_dict(), os.path.join(writer_dir, f"checkpoint.pt"))

    # if args.early_stopping:
    #     model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    

    # 여기서 final model 성능확인하는거 짜야됨 
    if args.model_type == 'clam_sb_multi' or args.model_type == 'clam_mb_multi': 
        balance = args.loss_balance
        output = {}
        
        # Final val err, auc
        _, final_val_error, final_val_error_subtype, final_val_auc, final_val_auc_subtype, \
            final_val_auc_tmb_each_subtype, _, _ = summary_multi(args, model, val_loader)
        print_info = [f'\nFinal Val ']
        if balance[0]:
            print_info.extend([f'err_tmb: {final_val_error:.4f}', f'AUC_tmb: {final_val_auc:.4f}'])
            output.update({'final_val_acc':1-final_val_error, 'final_val_auc':final_val_auc})
        if balance[1]:
            print_info.extend([f'err_sub: {final_val_error_subtype:.4f}', f'AUC_sub: {final_val_auc_subtype:.4f}'])
            output.update({'final_val_acc_subtype':1-final_val_error_subtype, 'final_val_auc_subtype':final_val_auc_subtype})        
        if final_val_auc_tmb_each_subtype is not None:
            val_auc_tmb_each =[f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' for i, auc_each in enumerate(final_val_auc_tmb_each_subtype)]
            print_info.extend(val_auc_tmb_each)
            output.update({f'final_val_auc_tmb_{list(args.label_dict2.keys())[i]}': f'{auc_each:.4f}' for i, auc_each in enumerate(final_val_auc_tmb_each_subtype)})
        print(f', '.join(print_info))

        # Final test err, auc
        results_dict, final_test_error, final_test_error_subtype, final_test_auc, final_test_auc_subtype, \
            final_test_auc_tmb_each_subtype, acc_logger, acc_logger_subtype = summary_multi(args, model, test_loader)
        print_info = [f'\nFinal Test ']
        if balance[0]:
            print_info.extend([f'err_tmb: {final_test_error:.4f}', f'AUC_tmb: {final_test_auc:.4f}'])
            output.update({'final_test_acc':1-final_test_error, 'final_test_auc':final_test_auc})
        if balance[1]:
            print_info.extend([f'err_sub: {final_test_error_subtype:.4f}', f'AUC_sub: {final_test_auc_subtype:.4f}'])
            output.update({'final_test_acc_subtype':1-final_test_error_subtype, 'final_test_auc_subtype':final_test_auc_subtype})
        if final_test_auc_tmb_each_subtype is not None:
            test_auc_tmb_each =[f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' for i, auc_each in enumerate(final_test_auc_tmb_each_subtype)]
            print_info.extend(test_auc_tmb_each)
            output.update({f'final_test_auc_tmb_{list(args.label_dict2.keys())[i]}': f'{auc_each:.4f}' for i, auc_each in enumerate(final_test_auc_tmb_each_subtype)})
        print(f', '.join(print_info))

        # Final classwise acc
        if balance[0]:
            for i in range(args.n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('\tFinal tmb class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer:
                    writer.add_scalar('final/test_tmb_class_{}_acc'.format(i), acc, 0)
        if balance[1]:
            for i in range(args.n_classes_subtype):
                acc, correct, count = acc_logger_subtype.get_summary(i)
                print('\tFinal subtype class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer:
                    writer.add_scalar('final/test_subtype_class_{}_acc'.format(i), acc, 0)
        # Final error, auc tensorboard 
        if writer:
            if balance[0]:
                writer.add_scalar('final/val_err_tmb', final_val_error, 0)
                writer.add_scalar('final/val_auc_tmb', final_val_auc, 0)
                writer.add_scalar('final/test_err_tmb', final_test_error, 0)
                writer.add_scalar('final/test_auc_tmb', final_test_auc, 0)
            if balance[1]:
                writer.add_scalar('final_sub/val_err_sub', final_val_error_subtype, 0)
                writer.add_scalar('final_sub/val_auc_sub', final_val_auc_subtype, 0)
                writer.add_scalar('final_sub/test_err_sub', final_test_error_subtype, 0)
                writer.add_scalar('final_sub/test_auc_sub', final_test_auc_subtype, 0)
                if final_val_auc_tmb_each_subtype is not None:
                    for subtype_name, auc_each in zip(list(args.label_dict2.keys()), final_val_auc_tmb_each_subtype):
                        writer.add_scalar(f'final/val_auc_{subtype_name}', auc_each, 0)
                    for subtype_name, auc_each in zip(list(args.label_dict2.keys()), final_test_auc_tmb_each_subtype):
                        writer.add_scalar(f'final/test_auc_{subtype_name}', auc_each, 0)
        
    else: 
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Final Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Final Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('Final class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)

    #################################### BEST MODEL TEST ####################################
    # early stopping 없애버리고 best validation auc 나왔던 model 불러서 test
    model.load_state_dict(best_model_state)
    if args.model_type == 'clam_sb_multi' or args.model_type == 'clam_mb_multi': 
        # Best val err, auc
        _, best_val_error, best_val_error_subtype, best_val_auc, best_val_auc_subtype, \
            best_val_auc_tmb_each_subtype, _, _ = summary_multi(args, model, val_loader)
        print_info = [f'\nBest Val ']
        if balance[0]:
            print_info.extend([f'err_tmb: {best_val_error:.4f}', f'AUC_tmb: {best_val_auc:.4f}'])
            output.update({'best_val_acc':1-best_val_error, 'best_val_auc':best_val_auc})
        if balance[1]:
            print_info.extend([f'err_sub: {best_val_error_subtype:.4f}', f'AUC_sub: {best_val_auc_subtype:.4f}'])
            output.update({'best_val_acc_subtype':1-best_val_error_subtype, 'best_val_auc_subtype':best_val_auc_subtype})        
        if best_val_auc_tmb_each_subtype is not None:
            val_auc_tmb_each =[f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' for i, auc_each in enumerate(best_val_auc_tmb_each_subtype)]
            print_info.extend(val_auc_tmb_each)
            output.update({f'best_val_auc_tmb_{list(args.label_dict2.keys())[i]}': f'{auc_each:.4f}' for i, auc_each in enumerate(best_val_auc_tmb_each_subtype)})
        print(f', '.join(print_info))

        # Best test err, auc
        results_dict, best_test_error, best_test_error_subtype, best_test_auc, best_test_auc_subtype, \
            best_test_auc_tmb_each_subtype, acc_logger, acc_logger_subtype = summary_multi(args, model, test_loader)
        print_info = [f'\nBest Test ']
        if balance[0]:
            print_info.extend([f'err_tmb: {best_test_error:.4f}', f'AUC_tmb: {best_test_auc:.4f}'])
            output.update({'best_test_acc':1-best_test_error, 'best_test_auc':best_test_auc})
        if balance[1]:
            print_info.extend([f'err_sub: {best_test_error_subtype:.4f}', f'AUC_sub: {best_test_auc_subtype:.4f}'])
            output.update({'best_test_acc_subtype':1-best_test_error_subtype, 'best_test_auc_subtype':best_test_auc_subtype})
        if best_test_auc_tmb_each_subtype is not None:
            test_auc_tmb_each =[f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' for i, auc_each in enumerate(best_test_auc_tmb_each_subtype)]
            print_info.extend(test_auc_tmb_each)
            output.update({f'best_test_auc_tmb_{list(args.label_dict2.keys())[i]}': f'{auc_each:.4f}' for i, auc_each in enumerate(best_test_auc_tmb_each_subtype)})
        print(f', '.join(print_info))

        # Best classwise acc
        if balance[0]:
            for i in range(args.n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('\tBest tmb class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer:
                    writer.add_scalar('best/test_tmb_class_{}_acc'.format(i), acc, 0)
        if balance[1]:
            for i in range(args.n_classes_subtype):
                acc, correct, count = acc_logger_subtype.get_summary(i)
                print('\tBest subtype class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer:
                    writer.add_scalar('best/test_subtype_class_{}_acc'.format(i), acc, 0)

        # Final val,test error,auc tensorboard 
        if writer:
            if balance[0]:
                writer.add_scalar('best/val_err_tmb', best_val_error, 0)
                writer.add_scalar('best/val_auc_tmb', best_val_auc, 0)
                writer.add_scalar('best/test_err_tmb', best_test_error, 0)
                writer.add_scalar('best/test_auc_tmb', best_test_auc, 0)
            if balance[1]:
                writer.add_scalar('best_sub/val_err_sub', best_val_error_subtype, 0)
                writer.add_scalar('best_sub/val_auc_sub', best_val_auc_subtype, 0)
                writer.add_scalar('best_sub/test_err_sub', best_test_error_subtype, 0)
                writer.add_scalar('best_sub/test_auc_sub', best_test_auc_subtype, 0)
                if best_val_auc_tmb_each_subtype is not None:
                    for subtype_name, auc_each in zip(list(args.label_dict2.keys()), best_val_auc_tmb_each_subtype):
                        writer.add_scalar(f'final/val_auc_{subtype_name}', auc_each, 0)
                    for subtype_name, auc_each in zip(list(args.label_dict2.keys()), best_test_auc_tmb_each_subtype):
                        writer.add_scalar(f'final/test_auc_{subtype_name}', auc_each, 0)
            writer.close()

        return results_dict, output

################################################ 완료 라인 ############################################

    else:    
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Best Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Best Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('Best class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('best/test_class_{}_acc'.format(i), acc, 0)

        if writer:
            writer.add_scalar('best/val_error', val_error, 0)
            writer.add_scalar('best/val_auc', val_auc, 0)
            writer.add_scalar('best/test_error', test_error, 0)
            writer.add_scalar('best/test_auc', test_auc, 0)
            writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
def train_loop_multi(
                epoch, 
                model, 
                loader, 
                optimizer, 
                n_classes, 
                n_classes_subtype, 
                writer = None, 
                loss_fn = None, 
                loss_fn_subtype = None, 
                balance=[1.,1.,1.]
                ):   
    '''
    balance가 bag_weight 역할까지 포함하여 tmb_loss, subtype_loss, instance_loss 3개 balance hyperparameter
    '''
    if balance[0] == 0 and balance[1] == 0:
        raise ValueError("Both balance[0] and balance[1] are zero. One of the value should bebigger than 0")
    if not all(0 <= x <= 1 for x in balance):
        raise ValueError("All values in the 'balance' list should be between 0 and 1.")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    if balance[0]:
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        train_loss = 0.
        train_error = 0.
    if balance[1]:
        acc_logger_subtype = Accuracy_Logger(n_classes=n_classes_subtype)
        train_loss_subtype = 0.
        train_error_subtype = 0.
    if balance[2]:
        acc_logger_instance = Accuracy_Logger(n_classes=n_classes if balance[0] else n_classes_subtype)
        train_loss_instance = 0.
        inst_count = 0

    print('\n')
    for batch_idx, (data, label, label_subtype) in enumerate(loader):
        data = data.to(device)
        if balance[0]:
            label = label.to(device)
        if balance[1]:
            label_subtype = label_subtype.to(device)

        logits, logits_subtype, _, instance_dict = model(data, label=label if balance[0] else label_subtype, instance_eval=balance[2]!=0)

        loss_total = 0
        if balance[0]:
            # acc
            Y_hat = torch.topk(logits.detach(), 1, dim=1)[1]     # [1,1]
            acc_logger.log(Y_hat, label)
            # loss
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            train_loss += loss_value
            loss_total += balance[0]*loss
            # error
            error = calculate_error(Y_hat, label)
            train_error += error

        if balance[1]:
            # acc
            Y_hat_subtype = torch.topk(logits_subtype.detach(), 1, dim=1)[1]     # [1,1]
            acc_logger_subtype.log(Y_hat_subtype, label_subtype)
            # loss
            loss_subtype = loss_fn_subtype(logits_subtype, label_subtype)
            loss_value_subtype = loss_subtype.item()
            train_loss_subtype += loss_value_subtype
            loss_total += balance[1]*loss_subtype   
            # err
            error_sub = calculate_error(Y_hat_subtype, label_subtype)
            train_error_subtype += error_sub

        if balance[2]:
            # acc
            inst_preds = instance_dict['inst_preds']        # numpy array
            inst_labels = instance_dict['inst_labels']      # numpy array
            acc_logger_instance.log_batch(inst_preds, inst_labels)
            # loss
            loss_instance = instance_dict['instance_loss']
            loss_value_instance = loss_instance.item()
            train_loss_instance += loss_value_instance
            loss_total += balance[2]*loss_instance 
            
            inst_count+=1

        L_tot = 0
        if (batch_idx + 1) % 20 == 0:
            if balance[0]:
                L_tot += loss_value
            if balance[1]:
                L_tot += loss_value_subtype
            if balance[2]:
                L_tot += loss_value_instance

            print_info = [f'batch {batch_idx}']
            if sum(1 for item in balance if item)>=2:
                print_info.append(f'L_tot: {L_tot:.4f}')
            if balance[0]:
                print_info.extend([f'L_tmb: {loss_value:.4f}', f'label: {label.item()}'])
            if balance[1]:
                print_info.extend([f'L_sub: {loss_value_subtype:.4f}', f'label_sub: {label_subtype.item()}'])
            if balance[2]:
                print_info.append(f'L_ins: {loss_value_instance:.4f}')
            print_info.append(f'bag_size: {data.size(0)}')
            print(f', '.join(print_info))

        # backward pass
        loss_total.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    if balance[0]:
        train_loss /= len(loader)
        train_error /= len(loader)
    if balance[1]:
        train_loss_subtype /= len(loader)
        train_error_subtype /= len(loader)
    if balance[2]:
        train_loss_instance /= inst_count
    print('\n')

    L_tot = 0
    if balance[0]:
        L_tot += train_loss
    if balance[1]:
        L_tot += train_loss_subtype
    if balance[2]:
        L_tot += train_loss_instance    

    # print loss & err
    print_info = [f'Epoch: {epoch}']
    if sum(1 for item in balance if item)>=2:
        print_info.append(f'L_tot: {L_tot:.4f}')
    if balance[0]:
        print_info.extend([f'L_tmb: {train_loss:.4f}', f'err_tmb: {train_error:.4f}'])
    if balance[1]:
        print_info.extend([f'L_sub: {train_loss_subtype:.4f}', f'err_subtype: {train_error_subtype:.4f}'])
    if balance[2]:
        print_info.append(f'L_ins: {train_loss_instance:.4f}')        
    print(f', '.join(print_info))

    # print classification acc
    if balance[0]:
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('\ttmb class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('train/tmb_class_{}_acc'.format(i), acc, epoch)
    if balance[1]:
        for i in range(n_classes_subtype):
            acc, correct, count = acc_logger_subtype.get_summary(i)
            print('\tsubtype class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('train/subtype_class_{}_acc'.format(i), acc, epoch)
    if balance[2]:
        for i in range(n_classes if balance[0] else n_classes_subtype):
            acc, correct, count = acc_logger_instance.get_summary(i)
            print(f'\t{"TMB" if balance[0] else "Subtype"} class {i} clustering: acc {acc} correct {correct}/{count}')
            # 얘는 안중요해서 저장 안하는 듯?
            # if writer:
            #     writer.add_scalar('train/subtype_class_{}_acc'.format(i), acc, epoch)

    if writer:
        # balance 중에서 2개 이상 true면 L_tot logging
        if sum(1 for item in balance if item)>=2:
            writer.add_scalar('train/L_tot', L_tot, epoch)
        if balance[0]:
            writer.add_scalar('train/L_tmb', train_loss, epoch)
            writer.add_scalar('train/error_tmb', train_error, epoch)
        if balance[1]:
            writer.add_scalar('train/L_subtype', train_loss_subtype, epoch)
            writer.add_scalar('train/error_subtype', train_error_subtype, epoch)
        if balance[2]:
            writer.add_scalar('train/L_instance', train_loss_instance, epoch)


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    scaler = GradScaler()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    end = time.time()   
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # if data.dtype == torch.float16:
        #     data = data.to(torch.float32)#, label.to(torch.int32)
        # data = data.to(torch.float16)
        # print(data.size())

        with autocast(enabled=True):    
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            loss = loss_fn(logits, label)
            instance_loss = instance_dict['instance_loss']
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
                
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        # measure elapsed time
        duration = time.time() - end
        end = time.time()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, time:{:.4f}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, duration, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        # total_loss.backward()
        scaler.scale(total_loss).backward()
        # step
        # optimizer.step()
        # optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        for param in model.parameters():
            param.grad = None

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
def validate_multi(
                args,
                cur, 
                epoch, 
                model, 
                loader, 
                early_stopping = None, 
                writer = None, 
                loss_fn = None, 
                loss_fn_subtype = None, 
                balance=[1.,0.,1.], 
                max_auc=0,
                ):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if balance[0]:
        acc_logger = Accuracy_Logger(n_classes=args.n_classes)
        val_loss = 0.
        val_error = 0.
        prob = np.zeros((len(loader), args.n_classes))
        labels = np.zeros(len(loader))
    if balance[1]:
        acc_logger_subtype = Accuracy_Logger(n_classes=args.n_classes_subtype)
        val_loss_subtype = 0.
        val_error_subtype = 0.
        prob_subtype = np.zeros((len(loader), args.n_classes_subtype))
        labels_subtype = np.zeros(len(loader))
    if balance[2]:
        acc_logger_instance = Accuracy_Logger(n_classes=args.n_classes if balance[0] else args.n_classes_subtype)
        val_loss_instance = 0.
        inst_count=0
    
    with torch.no_grad():
        for batch_idx, (data, label, label_subtype) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            if balance[0]:
                label = label.to(device, non_blocking=True)
            if balance[1]:
                label_subtype = label_subtype.to(device, non_blocking=True)

            logits, logits_subtype, _, instance_dict = model(data, label=label if balance[0] else label_subtype, instance_eval=balance[2]!=0)

            if balance[0]:
                # acc
                Y_hat = torch.topk(logits.detach(), 1, dim=1)[1]     # [1,1]
                Y_prob = F.softmax(logits, dim = 1)     # tmb prob
                acc_logger.log(Y_hat, label)
                # loss
                loss = loss_fn(logits, label)
                loss_value = loss.item()
                val_loss += loss_value
                # error, total loss
                error = calculate_error(Y_hat, label)
                val_error += error
                # auc 계산용
                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()
                
            if balance[1]:
                # acc
                Y_hat_subtype = torch.topk(logits_subtype.detach(), 1, dim=1)[1]     # [1,1]
                Y_prob_subtype = F.softmax(logits_subtype, dim = 1)     # subtype prob
                acc_logger_subtype.log(Y_hat_subtype, label_subtype)
                # loss
                loss_subtype = loss_fn_subtype(logits_subtype, label_subtype)
                loss_value_subtype = loss_subtype.item()
                val_loss_subtype += loss_value_subtype
                # auc 계산용
                prob_subtype[batch_idx] = Y_prob_subtype.cpu().numpy()
                labels_subtype[batch_idx] = label_subtype.item()
                # err, total loss
                error_sub = calculate_error(Y_hat_subtype, label_subtype)
                val_error_subtype += error_sub
            
            if balance[2]:
                # acc
                inst_preds = instance_dict['inst_preds']        # numpy array
                inst_labels = instance_dict['inst_labels']      # numpy array
                acc_logger_instance.log_batch(inst_preds, inst_labels)        
                # loss
                loss_instance = instance_dict['instance_loss']
                loss_value_instance = loss_instance.item()
                val_loss_instance += loss_value_instance

                inst_count+=1

    if balance[0]:
        val_loss /= len(loader)
        val_error /= len(loader)
    if balance[1]:
        val_loss_subtype /= len(loader)
        val_error_subtype /= len(loader)
    if balance[2]:
        val_loss_instance /= inst_count

    L_tot = 0
    if balance[0]:
        L_tot += val_loss
    if balance[1]:
        L_tot += val_loss_subtype
    if balance[2]:
        L_tot += val_loss_instance    
    
    # auc 계산
    if balance[0]:
        if args.n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')
    if balance[1]:
        if args.n_classes_subtype == 2:
            auc_subtype = roc_auc_score(labels_subtype, prob_subtype[:, 1])
        else:
            auc_subtype = roc_auc_score(labels_subtype, prob_subtype, multi_class='ovr')

    # tmb classification 하면서 class도 2개 이상인 경우 각 subtype 별로 따로 auc 구함
    if balance[0] and len(args.target_subtype)>1:
        max_subtype_label = int(np.max(labels_subtype)+1)
        auc_tmb_each_subtype = []
        # labels_subtype == i인 데이터만을 사용하여 tmb AUC 계산
        for i in range(max_subtype_label):
            mask_subtype_i = (labels_subtype == i)
            auc_subtype_i = roc_auc_score(labels[mask_subtype_i], prob[mask_subtype_i, 1])
            auc_tmb_each_subtype.append(auc_subtype_i)
        print_auc_tmb_each_subtype = [f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' \
                                      for i, auc_each in enumerate(auc_tmb_each_subtype)]
        # print_auc_tmb_each_subtype = ', '.join(print_auc_tmb_each_subtype)

    # max_flag 계산
    max_flag = False
    current_auc = auc if balance[0]>0 else auc_subtype
    if current_auc > max_auc:
        max_auc = current_auc
        max_flag = True

    # print loss & auc
    print_info = [f'\nVal Set']
    if sum(1 for item in balance if item)>=2:
        print_info.append(f'L_tot: {L_tot:.4f}')
    if balance[0]:
        print_info.extend([f'L_tmb: {val_loss:.4f}', f'err_tmb: {val_error:.4f}'])
    if balance[1]:
        print_info.extend([f'L_sub: {val_loss_subtype:.4f}', f'err_sub: {val_error_subtype:.4f}'])
    if balance[2]:
        print_info.extend([f'L_ins: {val_loss_instance:.4f}'])
    if balance[0]:
        print_info.extend([f'auc_tmb: {auc:.4f}'])
    if balance[1]:
        print_info.extend([f'auc_sub: {auc_subtype:.4f}'])
    if balance[0] and len(args.target_subtype)>1:
        print_info.extend(print_auc_tmb_each_subtype)
    print(f', '.join(print_info)) 

    # print classification accuracy
    if balance[0]:
        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print(f'\tTMB class {i}: acc {acc}, correct {correct}/{count}')     
    if balance[1]:
        for i in range(args.n_classes_subtype):
            acc, correct, count = acc_logger_subtype.get_summary(i)
            print(f'\tSubtype class {i}: acc {acc}, correct {correct}/{count}')             
    if balance[2]:
        for i in range(args.n_classes if balance[0] else args.n_classes_subtype):
            acc, correct, count = acc_logger_instance.get_summary(i)
            print(f'\t{"TMB" if balance[0] else "Subtype"} instance class {i}: acc {acc}, correct {correct}/{count}')             

    if writer:
        if sum(1 for item in balance if item)>=2:
            writer.add_scalar('val/L_tot', L_tot, epoch)
        if balance[0]:
            writer.add_scalar('val/L_tmb', val_loss, epoch)
            writer.add_scalar('val/error_tmb', val_error, epoch)
            writer.add_scalar('val/auc_tmb', auc, epoch)
            if len(args.target_subtype)>1:
                for i, auc_subtype_i in enumerate(auc_tmb_each_subtype):
                    writer.add_scalar(f'val/auc_{list(args.label_dict2.keys())[i]}', auc_subtype_i, epoch)
        if balance[1]:
            writer.add_scalar('val/L_subtype', val_loss_subtype, epoch)
            writer.add_scalar('val/error_subtype', val_error_subtype, epoch)
            writer.add_scalar('val/auc_subtype', auc_subtype, epoch)
        if balance[2]:
            writer.add_scalar('val/L_instance', val_loss_instance, epoch)

    # if early_stopping:
    #     assert args.results_dir
    #     early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         return True, max_flag, max_auc

    return False, max_flag, max_auc


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, max_auc=0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if data.dtype == torch.float16:
                data = data.to(torch.float32)#, label.to(torch.int32)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    max_flag = False
    if auc > max_auc:
        max_auc = auc
        max_flag = True
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, max_flag, max_auc

    return False, max_flag, max_auc

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, max_auc=0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            if data.dtype == torch.float16:
                data = data.to(torch.float32)#, label.to(torch.int32)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    
    max_flag = False
    if auc > max_auc:
        max_auc = auc
        max_flag = True

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, max_flag, max_auc

    return False, max_flag, max_auc

def summary_multi(args, model, loader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    balance = args.loss_balance

    # return 하는 변수들
    test_error = None
    test_error_subtype = None
    auc = None
    auc_subtype = None
    auc_tmb_each_subtype = None
    acc_logger = None
    acc_logger_subtype = None

    if balance[0]:
        acc_logger = Accuracy_Logger(n_classes=args.n_classes)
        test_error = 0.
        prob = np.zeros((len(loader), args.n_classes))
        labels = np.zeros(len(loader))
    if balance[1]:
        acc_logger_subtype = Accuracy_Logger(n_classes=args.n_classes_subtype)
        test_error_subtype = 0.
        prob_subtype = np.zeros((len(loader), args.n_classes_subtype))
        labels_subtype = np.zeros(len(loader))
    if balance[2]:
        acc_logger_instance = Accuracy_Logger(n_classes=args.n_classes if balance[0] else args.n_classes_subtype)
        test_loss_instance = 0.
        inst_count=0

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, label_subtype) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        if balance[0]:
            label = label.to(device, non_blocking=True)
        if balance[1]:
            label_subtype = label_subtype.to(device)
                
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, logits_subtype, _, instance_dict = model(data, label=label if balance[0] else label_subtype, 
                                                             instance_eval=balance[2]!=0)

        patient_results.update({slide_id: {'slide_id': np.array(slide_id)}})
        if balance[0]:
            # acc
            Y_hat = torch.topk(logits.detach(), 1, dim=1)[1]     # [1,1]
            Y_prob = F.softmax(logits, dim = 1)     # tmb prob
            acc_logger.log(Y_hat, label)
            # auc 계산용        
            probs = Y_prob.cpu().numpy()        
            prob[batch_idx] = probs
            labels[batch_idx] = label.item()
            # error
            error = calculate_error(Y_hat, label)
            test_error += error
            # result for save pickle
            # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob_sub': probs_sub, 'label_sub': label_subtype.item()}})  # subtype 정보는 저장 안함
            patient_results[slide_id].update({'prob': probs, 'label': label.item()})  # subtype 정보는 저장 안함
        if balance[1]:
            # acc
            Y_hat_subtype = torch.topk(logits_subtype.detach(), 1, dim=1)[1]     # [1,1]
            Y_prob_subtype = F.softmax(logits_subtype, dim = 1)     # subtype prob
            acc_logger_subtype.log(Y_hat_subtype, label_subtype)
            # auc 계산용
            probs_sub = Y_prob_subtype.cpu().numpy()
            prob_subtype[batch_idx] = probs_sub
            labels_subtype[batch_idx] = label_subtype.item()
            # err, total loss
            error_sub = calculate_error(Y_hat_subtype, label_subtype)
            test_error_subtype += error_sub
            # result for save pickle
            # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob_sub': probs_sub, 'label_sub': label_subtype.item()}})  # subtype 정보는 저장 안함
            patient_results[slide_id].update({'prob_sub': probs_sub, 'label_sub': label_subtype.item()})  # subtype 정보는 저장 안함
        if balance[2]:
            # acc
            inst_preds = instance_dict['inst_preds']        # numpy array
            inst_labels = instance_dict['inst_labels']      # numpy array
            acc_logger_instance.log_batch(inst_preds, inst_labels)        
            # loss
            loss_instance = instance_dict['instance_loss']
            loss_value_instance = loss_instance.item()
            test_loss_instance += loss_value_instance

            inst_count+=1

    
    if balance[0]:
        test_error /= len(loader)
    if balance[1]:
        test_error_subtype /= len(loader)
    if balance[2]:
        test_loss_instance /= inst_count

    # auc 계산
    # if args.n_classes == 2:
    #     auc = roc_auc_score(labels, prob[:, 1])
    #     if balance[1]:
    #         auc_subtype = roc_auc_score(labels_subtype, prob_subtype[:, 1])
    #     aucs = []
    # else:
    #     aucs = []
    #     aucs_subtype = []
    #     binary_labels = label_binarize(labels, classes=[i for i in range(args.n_classes)])
    #     if balance[1]:
    #         binary_labels_subtype = label_binarize(labels_subtype, classes=[i for i in range(args.n_classes)])
    #     for class_idx in range(args.n_classes):
    #         if class_idx in labels:
    #             fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
    #             aucs.append(calc_auc(fpr, tpr))
    #             if balance[1]:        
    #                 fpr, tpr, _ = roc_curve(binary_labels_subtype[:, class_idx], prob_subtype[:, class_idx])
    #                 aucs_subtype.append(calc_auc(fpr, tpr))
    #         else:
    #             aucs.append(float('nan'))
    #             if balance[1]:
    #                 aucs_subtype.append(float('nan'))

    #     auc = np.nanmean(np.array(aucs))
    #     if balance[1]:    
    #         auc_subtype = np.nanmean(np.array(aucs_subtype))
    if balance[0]:
        if args.n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=[i for i in range(args.n_classes)])            
            for class_idx in range(args.n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc = np.nanmean(np.array(aucs))
    if balance[1]:
        if args.n_classes_subtype == 2:
            auc_subtype = roc_auc_score(labels_subtype, prob_subtype[:, 1])
        else:
            aucs_subtype = []
            binary_labels_subtype = label_binarize(labels_subtype, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes_subtype):
                if class_idx in labels_subtype:
                    fpr, tpr, _ = roc_curve(binary_labels_subtype[:, class_idx], prob_subtype[:, class_idx])
                    aucs_subtype.append(calc_auc(fpr, tpr))
                else:
                    aucs_subtype.append(float('nan'))
            auc_subtype = np.nanmean(np.array(aucs_subtype))

    # tmb classification 하면서 class도 2개 이상인 경우 각 subtype 별로 따로 auc 구함
    if balance[0] and len(args.target_subtype)>1:
        max_subtype_label = int(np.max(labels_subtype)+1)
        auc_tmb_each_subtype = []
        # labels_subtype == i인 데이터만을 사용하여 tmb AUC 계산
        for i in range(max_subtype_label):
            mask_subtype_i = (labels_subtype == i)
            auc_subtype_i = roc_auc_score(labels[mask_subtype_i], prob[mask_subtype_i, 1])
            auc_tmb_each_subtype.append(auc_subtype_i)
        # print_auc_tmb_each_subtype = [f'auc_tmb_{list(args.label_dict2.keys())[i]}: {auc_each:.4f}' \
        #                               for i, auc_each in enumerate(auc_tmb_each_subtype)]
    # if balance[1]:
    #     # labels_subtype == 0인 데이터만을 사용하여 AUC 계산
    #     mask_subtype_0 = (labels_subtype == 0)
    #     auc_subtype_0 = roc_auc_score(labels[mask_subtype_0], prob[mask_subtype_0, 1])

    #     # labels_subtype == 1인 데이터만을 사용하여 AUC 계산
    #     mask_subtype_1 = (labels_subtype == 1)
    #     auc_subtype_1 = roc_auc_score(labels[mask_subtype_1], prob[mask_subtype_1, 1]) 

    return patient_results, test_error, test_error_subtype, auc, auc_subtype, \
        auc_tmb_each_subtype, acc_logger, acc_logger_subtype


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if data.dtype == torch.float16:
                data = data.to(torch.float32)#, label.to(torch.int32)
                
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
