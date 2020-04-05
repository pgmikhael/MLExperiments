import torch
import os
import torch.nn as nn
from copy import copy
"""
When saving a general checkpoint, to be used for either inference or resuming training, 
    you must save more than just the model’s state_dict. It is important to also save the optimizer’s 
    state_dict, as this contains buffers and parameters that are updated as the model trains. 
    Other items that you may want to save are the epoch you left off on, the latest recorded training 
    loss, external torch.nn.Embedding layers, etc.

To save multiple components, organize them in a dictionary and use torch.save() to serialize the 
    dictionary. A common PyTorch convention is to save these checkpoints using the .tar file extension.

To load the items, first initialize the model and optimizer, then load the dictionary locally using 
    torch.load(). From here, you can easily access the saved items by simply querying the dictionary 
    as you would expect.

Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation 
    mode before running inference. Failing to do this will yield inconsistent inference results. 
    If you wish to resuming training, call model.train() to ensure these layers are in training mode.

When loading a model on a CPU that was trained with a GPU, pass torch.device('cpu') to the map_location 
    argument in the torch.load() function. In this case, the storages underlying the tensors are 
    dynamically remapped to the CPU device using the map_location argument.

    Make sure to call input = input.to(device) on any input tensors that you feed to the model
    
            device = torch.device("cuda")
            model = TheModelClass(*args, **kwargs)
            model.load_state_dict(torch.load(PATH))
            model.to(device)
"""

MODEL_REGISTRY = {}
NO_OPTIM_ERR  = "{} Optimizer Not Found"

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(args):
    model = MODEL_REGISTRY[args.model_name](args)

    if args.data_parallel:
        model = nn.DataParallel(model)

    if args.cuda:
        model.to(args.device)
        
    optimizer = get_optimizer(model, args)

    return model, optimizer

def load_model(path, model, optimizer, args):
    if args.results_path is not None:
        path = "{}_model.pt".format(args.results_path)
        
    print('\nLoading model from {}'.format(path))
    try:
        # if not args.cuda:
        #     checkpoint = torch.load(path, map_location = torch.device('cpu'))
        # else:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.data_parallel:
            model = nn.DataParallel(model)

        if args.cuda:
            model.to(args.device)


        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_stats = checkpoint['epoch_stats']
        lr = checkpoint['lr']
    except:
        raise Exception("Snapshot {} does not exist!".format(path))
    
    return model, optimizer, lr, epoch_stats

def save_model(model, optimizer, epoch_stats, args):
    if args.results_path is not None:
        path = "{}_model.pt".format(args.results_path)
    else:
        path = os.path.join(args.save_dir, "{}_{}_{}_model.pt".format(args.model_name, \
        args.dataset, args.run_time))

    torch.save({
            #'model': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_stats': epoch_stats,
            'lr': args.lr
            }, path)

def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay= args.weight_decay)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(params,lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, lr = args.lr, momentum=args.momentum, weight_decay= args.weight_decay)
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))


def get_rolled_out_size(args):
    h, w = args.img_size
    dummy_input = torch.ones(1, 3, h, w)
    model = get_model(args)[0]
    dummy_output = model.forward_thru_convs(dummy_input)
    return dummy_output.shape[-1]
