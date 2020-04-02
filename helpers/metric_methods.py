import sklearn.metrics
import warnings
from collections import defaultdict
import torch.nn as nn

def init_metrics_dictionary(mode):
    '''
    Return empty metrics dict
    '''
    stats_dict = defaultdict(list)
    stats_dict['best_epoch'] = 0
    return stats_dict

def compute_eval_metrics(golds, preds, probs, loss, args, stats_dict, mode):
    stats_dict['{}_loss'.format(mode)].append(loss)
    stats_dict['{}_preds'.format(mode)] = preds
    stats_dict['{}_golds'.format(mode)] = golds
    stats_dict['{}_probs'.format(mode)] = probs

    if 'regression' in args.dataset:
        return regression_metrics(golds, preds, probs, loss, args, stats_dict, mode)
    else:
        return classification_metrics(golds, preds, probs, loss, args, stats_dict, mode)


def classification_metrics(golds, preds, probs, loss, args, stats_dict, mode):
    accuracy = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
    precision = sklearn.metrics.precision_score(y_true=golds, y_pred=preds)
    recall = sklearn.metrics.recall_score(y_true=golds, y_pred=preds)
    f1 = sklearn.metrics.f1_score(y_true=golds, y_pred=preds)
    confusion_matrix = sklearn.metrics.confusion_matrix(golds, preds) 
    try:
        auc = sklearn.metrics.roc_auc_score(golds, probs, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate metrics because {}".format(e))
        auc = 'NA'

    stats_dict['{}_accuracy'.format(mode)].append(accuracy)
    stats_dict['{}_precision'.format(mode)].append(precision)
    stats_dict['{}_recall'.format(mode)].append(recall)
    stats_dict['{}_f1'.format(mode)].append(f1)
    stats_dict['{}_auc'.format(mode)].append(auc)
    stats_dict['{}_confusion_matrix'.format(mode)].append(confusion_matrix.tolist())

    log_statement = '\n{} Phase Stats\n --loss: {} acc: {} auc: {} (m={}, p={}), precision: {} recall: {} f1: {}'.format(
       mode.upper(), loss, accuracy, auc, len(golds), sum(golds), precision, recall, f1)
   
    return log_statement, stats_dict

def regression_metrics(golds, preds, probs, loss, args, stats_dict, mode):
    mse = sklearn.metrics.mean_squared_error(y_true=golds, y_pred=probs)
    mae = sklearn.metrics.mean_absolute_error(y_true=golds, y_pred=probs)
    r2_score = sklearn.metrics.r2_score(y_true=golds, y_pred=probs)

    stats_dict['{}_mse'.format(mode)].append(mse)
    stats_dict['{}_mae'.format(mode)].append(mae)
    stats_dict['{}_r2'.format(mode)].append(r2_score)

    log_statement = '\n{} Phase Stats\n --loss: {} mse: {} mae: {} r2: {} '.format(
       mode.upper(), loss, mse, mae, r2_score)

    return log_statement, stats_dict


def get_criterion(args):
    if args.criterion == 'binary_cross_entropy':
        return nn.BCEWithLogitsLoss()
    elif args.criterion == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif args.criterion == 'l1_loss':
        return nn.L1Loss()
    elif args.criterion == 'mse_loss': 
        return nn.MSELoss()
    elif args.criterion == 'smooth_l1_loss': 
        return nn.SmoothL1Loss()