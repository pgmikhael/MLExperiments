"""
Command-Line Arguments
"""
import argparse
import itertools
import torch
from datasets.dataset_factory import get_dataset_class

POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'


def parse_args():
     parser = argparse.ArgumentParser(description='MLPIPE')
     parser.add_argument('--oauth2_path', type = str, default = 'secrets/oauth_cred.json')

     # Dataset
     ## Development
     parser.add_argument('--dataset', type = str, default = 'regression')
     parser.add_argument('--data_dir', type = str, default = 'Data/')
     parser.add_argument('--img_dir', type = str, default = 'Data/pngs/')
     parser.add_argument('--json_dataset', type = str, default = 'ig_cnn')
     parser.add_argument('--task', type = str, default = 'likes')

     # Model
     parser.add_argument('--model_name', type = str, default = 'vgg', help = 'Name of DNN')
     parser.add_argument('--trained_on_imagenet', action='store_true', default = False, help = 'torchvision weights from imagenet trained models')
     parser.add_argument('--clip_supp_weights', action='store_true', default = False, help = 'clip weights of suppressive field to nonegative range')

     # Transformers
     parser.add_argument('--train_img_transformers', nargs='*', default=['scale_2d'], help='List of image-transformations to use [default: ["scale_2d"]] \
                        Usage: "--image_transformers trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"')
     parser.add_argument('--train_tnsr_transformers', nargs='*', default=['force_num_chan'], help='List of image-transformations to use [default: ["scale_2d"]] \
                        Usage: "--image_transformers trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"')
     parser.add_argument('--test_img_transformers', nargs='*', default=['scale_2d'], help='List of image-transformations to use for the dev and test dataset [default: ["scale_2d"]]')
     parser.add_argument('--test_tnsr_transformers', nargs='*', default=['force_num_chan'], help='List of image-transformations to use for the dev and test dataset [default: ["force_num_chan"]]')
     parser.add_argument('--num_chan', type = int, default=3)

     # Dataset stats
     parser.add_argument('--get_dataset_stats', action='store_true', default = False, help = 'whether to get mean, std of training set')
     parser.add_argument('--img_size',  type=int, nargs='+', default=[256, 256], help='width and height of image in pixels. [default: [256,256]')
     parser.add_argument('--img_mean', type=float, nargs='+', default=[0.4979, 0.4665, 0.4492], help='mean value of img pixels. Per channel. ')
     parser.add_argument('--img_std', type=float, nargs='+', default=[0.2853, 0.2678, 0.2685 ], help='std of img pixels. Per channel. ')

     # Model Output Sizes
     parser.add_argument('--forward_thru_convs', action='store_true', default = False, help = 'whether training model')
     

     # Workers and GPUS
     parser.add_argument('--num_workers', type=int, default = 0, help='std of img pixels. Per channel. ')
    
     # Learning
     parser.add_argument('--train_phase', action='store_true', default = False, help = 'whether training model')
     parser.add_argument('--test_phase', action='store_true', default = False, help = 'whether testing model')
     parser.add_argument('--resume', action='store_true', default = False, help = 'whether to resume from previous run')
     parser.add_argument('--init_lr', type = float, default = 0.0001, help = 'learning rate')
     parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
     parser.add_argument('--lr_decay', type = float, default = 1, help = 'how much to reduce lr by when getting closer to optimum')
     parser.add_argument('--patience', type = float, default= 10, help = 'how much to wait before reducing lr')
     parser.add_argument('--momentum', type = float, default = 0.99, help = 'optimizer momentum')
     parser.add_argument('--weight_decay', type = float, default = 1, help = 'l2 penalty coefficient')
     parser.add_argument('--l1_decay', type = float, default = 0.1, help = 'l1 penalty coefficient')
     parser.add_argument('--dropout', type = float, default = 0.25, help = 'dropout probability')
     parser.add_argument('--use_dropout', action = 'store_true', default = False, help = 'if using dropout ')
     parser.add_argument('--num_epochs', type = int, default = 100, help = 'number of epochs')
     parser.add_argument('--batch_size', type = int, default = 10, help = 'batch size')
     parser.add_argument('--burn_in', type = int, default = 10, help = 'number of epochs before saving improved model')
     parser.add_argument('--optimizer', type = str, default = 'adam', help = 'optimizer function')
     parser.add_argument('--criterion', type = str, default = 'binary_cross_entropy', help = 'optimizer function')
     parser.add_argument('--tuning_metric', type = str, default= 'dev_loss', help = 'metric on which to tune model')
     
     # Inference
     parser.add_argument('--plot_losses', action='store_true', default = False, help = 'whether to plot losses')
     parser.add_argument('--plot_accuracies', action='store_true', default = False, help = 'whether to plot accuracies')

     # Directories and Files
     parser.add_argument('--viz_path', type = str, help = 'path to save visualizations')
     parser.add_argument('--snapshot_path', type = str, help = 'path to snapshot if using saved model')
     parser.add_argument('--results_dir', type = str, default = '/Users/petermikhael/pgmikhael.github.io/ig_models_results', help = 'path to results dir')
     parser.add_argument('--save_dir', type = str, default = '/Users/petermikhael/pgmikhael.github.io/ig_models_results', help = 'directory of models')
     parser.add_argument('--results_path', type = str, help = 'defined either automatically by dispatcher.py or time in main.py. Keep without default')

     # CUDA
     parser.add_argument('--cuda', action='store_true', default = False, help = 'whether to use gpu')
     parser.add_argument('--data_parallel', action = 'store_true', default = False, help = 'whether use data parallel')
     args = parser.parse_args()
     
     # Set args particular to dataset
     get_dataset_class(args).set_args(args)

     # define if cuda device
     args.cuda = args.cuda and torch.cuda.is_available()
     args.device = 'cuda' if args.cuda else 'cpu'
     args.lr = args.init_lr

     # Compute stats skips normalization in transforms
     args.computing_stats = False
     return args


def parse_dispatcher_config(config):
     '''
     Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
     but the config contains a list, this will return one job for each item in the list.
     :config - experiment_config

     returns: jobs - a list of flag strings, each of which encapsulates one job.
          *Example: --train --cuda --dropout=0.1 ...
     returns: experiment_axies - axies that the grid search is searching over
     '''

     search_spaces = config['search_space']
     flags = []
     arguments = []
     experiment_axies = []

     # add anything outside search space as fixed
     fixed_args = ""
     for arg in config: 
          if arg != 'search_space' and arg != 'available_gpus':
               if type(config[arg]) is bool:
                    if config[arg]:
                         fixed_args += '--{} '.format(str(arg))
                    else:
                         continue
               else:
                    fixed_args += '--{} {} '.format(arg, config[arg])

     for key, value in search_spaces.items():
          flags.append(key)
          arguments.append(value)
          if len(value) > 1:
               experiment_axies.append(key)

     experiments = []
     exps_combs = list(itertools.product(*arguments))

     for tpl in exps_combs:
          exp = ""
          for idx, flg in enumerate(flags):
               if type(tpl[idx]) is bool:
                    if tpl[idx]:
                         exp += '--{} '.format(str(flg))
                    else:
                         continue
               else:
                    exp += '--{} {} '.format(str(flg), str(tpl[idx]))
          exp += fixed_args
          experiments.append(exp)
     
     return experiments, flags, experiment_axies