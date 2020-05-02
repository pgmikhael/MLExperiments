import sys
from os.path import dirname, realpath
sys.path.append((dirname(dirname(realpath(__file__)))))
import argparse
import subprocess
import os
import multiprocessing
import pickle
import csv
import json
import random
import helpers.parsing as parsing
import hashlib
from helpers.reporting_methods import yagmail_results, email_results
from collections import defaultdict
import pandas as pd

EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
SUCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}."
RESULT_KEY_STEMS = ['{}_loss', '{}_accuracy', '{}_auc', '{}_precision', '{}_recall', '{}_f1', '{}_mse', '{}_mae', '{}_r2']

LOG_KEYS = ['results_path', 'model_path', 'log_path']
SORT_KEY = 'dev_loss'

parser = argparse.ArgumentParser(description='NAB DNN Dispatcher. For use information, see `doc/README.md`')
parser.add_argument('--config_path', type = str, required = True, default = '/Users/petermikhael/pgmikhael.github.io/nab/configs/nba_configs.json', help = 'path to model configurations json file')
parser.add_argument('--log_dir', type=str, default="/Users/petermikhael/pgmikhael.github.io/nab_models_results/nab_grid_search", help="path to store logs and detailed job level result files")
parser.add_argument('--result_path', type=str, default="/grid_search.csv", help="path to store grid_search table.")


def launch_experiment(gpu, flag_string):
    '''
    DONE.
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.rslt".format(log_stem)

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u main.py {} --results_path {}".format(
        gpu, flag_string, log_stem) #use log_stem instead of results_path, add extensions in main/learn.py

    # forward logs to logfile
    if "--resume" in flag_string:
        pipe_str = ">>"
    else:
        pipe_str = ">"

    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)
    print("Launched exp: {}".format(shell_cmd))

    if not os.path.exists(results_path):
        subprocess.call(shell_cmd, shell=True)

    return results_path, log_path

def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def worker(gpu, job_queue, done_queue): 
    '''
    DONE.
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(launch_experiment(gpu, params))
        
def update_sumary_with_results(result_path, log_path, experiment_axies,  summary):
    assert result_path is not None
    try:
        result_dict = pickle.load(open(result_path, 'rb'))
        dict_path = result_path.split(".")[0]+".args"
        args_dict = pickle.load(open(dict_path, 'rb'))
    except Exception:
        print("Experiment failed! Logs are located at: {}".format(log_path))
        return summary

    summary['result_path'].append(result_path)
    summary['log_path'].append(log_path)
    summary['model_path'].append(result_dict['train_stats']['model_path'])

    best_epoch_indx = result_dict['train_stats']['best_epoch'] if result_dict['train_stats'] else 0

    for key in result_keys:
        if args_dict['test_phase']:
            for mode in ['test', 'dev']:
                if (key in result_dict['{}_stats'.format(mode)]) and len(result_dict['{}_stats'.format(mode)][key])>0:
                    summary[key].append(result_dict['{}_stats'.format(mode)][key][-1])
        if args_dict['train_phase']:
            if (key in result_dict['train_stats']) and (len(result_dict['train_stats'][key])>0):
                if ('dev' in key) and (not args_dict['test_phase']):
                        summary[key].append( result_dict['train_stats'][key][best_epoch_indx] )
                if ('train' in key):
                    summary[key].append( result_dict['train_stats'][key][best_epoch_indx] )

    for key in experiment_axies:
        if key in args_dict:
            summary[key].append(args_dict[key])
        else:
            summary[key].append('NA')
    
    result_dir = os.path.dirname(args.result_path)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    summary_csv = pd.DataFrame(summary)
    if SORT_KEY in summary_csv.columns:
        summary_csv.sort_values(SORT_KEY, axis = 0, ascending = False, inplace = True)
    summary_csv.to_csv(args.result_path, index = False)
    return summary

if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.config_path, 'r'))
    
    experiments, flags, experiment_axies = parsing.parse_dispatcher_config(experiment_config)


    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in experiments:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(experiments)))
    print()

    if experiment_config['cuda']:
        for gpu in experiment_config['available_gpus']:
            print("Start gpu worker {}".format(gpu))
            multiprocessing.Process(target=worker, args=(gpu, job_queue, done_queue)).start()
            print()
    
    result_keys = []
    for mode in ['train','dev','test']:
        result_keys.extend( [k.format(mode) for k in RESULT_KEY_STEMS ])

    summary = defaultdict(list)
    for i in range(len(experiments)):
        result_path, log_path = done_queue.get() #.rslt and .txt (stderr/out) files
        summary = update_sumary_with_results(result_path, log_path, experiment_axies, summary)
        dump_result_string = SUCESSFUL_SEARCH_STR.format(args.result_path)
        print("({}/{}) \t {}".format(i+1, len(experiments), dump_result_string))
    
    yagmail_results(args.result_path)
