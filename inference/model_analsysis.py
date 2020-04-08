import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import os

parser = argparse.ArgumentParser(description='. For use information, see `doc/README.md`')
parser.add_argument('--viz_dir', type = str, required = True, default = 'results/viz/', help = 'directory with visualizations saved')
parser.add_argument('--plot_metrics', action = 'store_true', default= False, help="whether to plot metrics")
parser.add_argument('--results_paths', type=str, nargs='+', default= ["results/viz/result1.rslt", "results/viz/result2.rslt"], help="path to result files.")
parser.add_argument('--model_names', type=str, nargs='+', default=['model1', 'model2'], help="model names to be compared.")

COLORS = ['blue', 'green', 'red']

def plot_metrics(metric, paths, model_names, viz_dir):
    for mode in ['train', 'dev']:
        img_path = '{}_{}.png'.format(mode, metric)
        img_path = os.path.join(viz_dir, img_path)
        fig = plt.figure()
        ax = plt.axes()
        legend = []
        for idx, path in enumerate(paths):
            legend.append(model_names[idx])
            results = pickle.load(open(path, 'rb' ))
            y = results['train_stats']['{}_{}'.format(mode, metric)]
            epochs = np.arange(len(y))
            ax.plot(epochs, y, color= COLORS[idx])
        ax.legend(legend)
        fig.savefig(img_path)

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.isdir(args.viz_dir):
        os.makedirs(args.viz_dir)
    
    if args.plot_metrics:
        plot_metrics('loss', args.paths, args.model_names, args.viz_dir)
        plot_metrics('accuracy', args.paths, args.model_names, args.viz_dir )