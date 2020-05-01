from collections import defaultdict
import os
from tqdm import tqdm
import torch.nn as nn
from helpers.metric_methods import compute_eval_metrics, get_criterion, init_metrics_dictionary
import warnings
import torch
from torch.utils import data
from models.model_factory import save_model, load_model
import numpy as np
import pdb

def run_model(x, y, batch, model, optimizer, crit, mode, args):     
    probs = model(x)
    B, C = probs.shape
    if C > 1:
        loss = crit(probs, y)    # compute loss
        preds = torch.softmax(probs, dim = -1)
        probs, preds = torch.topk(preds, k = 1)
        probs, preds = probs.view(B), preds.view(B)
    else:
        loss = crit(probs, y.unsqueeze(1).float())    # compute loss
        preds = (torch.sigmoid(probs) > 0.5)

    if args.l1_decay > 0:
        loss += args.l1_decay*l1_regularization(model)
    return loss, preds, probs, y

def epoch_pass(data_loader, model, optimizer, crit, mode, args):
    
    preds = []
    probs = []
    golds = []
    losses = []
    batch_loss = 0

    if mode == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()


    #i = 0
    temperature = args.init_temperature
    with tqdm(data_loader, total = len(data_loader), ncols = 60) as tqdm_bar:#total=args.num_batches_per_epoch)
        for batch in data_loader:
            x, y, batch = prepare_batch(batch, args)

            if batch is None:
                warnings.warn('Empty batch')
                continue
            
            loss, batch_preds, batch_probs, batch_golds = run_model(x.float(), y, batch, model, optimizer, crit, mode, args)

            batch_loss = loss.cpu().data.item()

            if mode == 'train':
                loss.backward()       
                optimizer.step()      
                optimizer.zero_grad() 

                temperature -= args.temperature_lr
                for name, param in model.named_parameters():
                    if not args.scale_biases:
                        if 'weight' in name:
                            param.data /= temperature
                        else:
                            continue
                    else:
                        param.data /= temperature     

            if args.cuda:
                losses.append(batch_loss)
                preds.extend(batch_preds.cpu().detach().numpy())
                probs.extend(batch_probs.cpu().detach().numpy())
                golds.extend(batch_golds.cpu().detach().numpy())
            else:
                losses.append(batch_loss)
                preds.extend(batch_preds.detach().numpy())
                probs.extend(batch_probs.detach().numpy())
                golds.extend(batch_golds.detach().numpy())

            i+=1
            tqdm_bar.update()
            # if i > args.num_batches_per_epoch:
            #     data_iter.__del__()
            #     break
    
    # format golds, preds, probs, avg_loss
    avg_loss = np.mean(np.array(losses))
    preds = np.array(preds)
    probs = np.array(probs)
    golds = np.array(golds)

    return golds, preds, probs, avg_loss


def train_model(train_data, dev_data, model, optimizer, args):

    if args.epoch_stats is not None:
        epoch_stats = args.epoch_stats
        start_epoch = epoch_stats['best_epoch'] + 1
    else:
        epoch_stats = init_metrics_dictionary(['train', 'dev'])
        start_epoch = 0

    crit = get_criterion(args)

    train_data_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True)

    dev_data_loader = data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True)

    num_epoch_sans_improvement = 0
    if args.cuda:
        DEVICE = torch.device("cuda")
        model.to(DEVICE)

    for epoch in range(start_epoch, args.num_epochs):
        if (epoch+1)%args.burn_in == 0:
            print("-------------\nEpoch {}:\n".format(epoch+1))
        for mode, data_loader in [('train', train_data_loader), ('dev', dev_data_loader)]:
            golds, preds, probs, loss = epoch_pass(data_loader, model, optimizer, crit, mode, args)

            log_statement, epoch_stats = compute_eval_metrics(golds, preds, probs, loss, args, epoch_stats, mode)
            
            if (epoch+1)%args.burn_in == 0:
                print(log_statement)

        # Save model if beats best dev
        best_func, arg_best = (min, np.argmin) if args.tuning_metric in ['dev_loss', 'dev_mse', 'dev_mae'] else (max, np.argmax)


        if (epoch+1)%args.burn_in == 0:
            improved = best_func(epoch_stats[args.tuning_metric]) == epoch_stats[args.tuning_metric][-args.burn_in]
            if improved:
                num_epoch_sans_improvement = 0
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                
                #assert epoch == arg_best( epoch_stats[args.tuning_metric] )
                epoch_stats['best_epoch'] = arg_best( epoch_stats[args.tuning_metric] )
                
                save_model(model, optimizer, epoch_stats, args)
                
            else:
                num_epoch_sans_improvement += 1*args.burn_in
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
                    args.lr *= args.lr_decay

            print('\n---- Best {} is {} at epoch {} ----'.format(
                args.tuning_metric,
                epoch_stats[args.tuning_metric][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))

    best_model_path = os.path.join(args.save_dir, "{}_{}_{}_model.pt".format( \
        args.model_name, args.dataset, args.run_time) )
        
    if os.path.isfile(best_model_path):
        model, optimizer, _, _ = load_model(best_model_path, model, optimizer, args)

    epoch_stats['model_path'] = best_model_path
    return epoch_stats, model, optimizer


def eval_model(eval_data, model, optimizer, mode, args):
    
    epoch_stats = init_metrics_dictionary(mode)

    data_loader = data.DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    if args.cuda:
        DEVICE = torch.device("cuda")
        model.to(DEVICE)

    crit = get_criterion(args)

    golds, preds, probs, loss = epoch_pass(data_loader, model, optimizer, crit, mode, args)

    log_statement, epoch_stats = compute_eval_metrics(golds, preds, probs, loss, args, epoch_stats, mode)
    
    print(log_statement)

    return epoch_stats


def l1_regularization(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.norm(param, p=1) # torch.sum(torch.abs(param))
    return regularization_loss

def prepare_batch(batch, args):
    x, y = batch['x'], batch['y']
    if args.cuda:
        x, y = x.to(args.device), y.to(args.device)
    return x, y, batch