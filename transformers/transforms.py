import torch
import torchvision

EMPTY_NAME_ERR = 'Name of transformer or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'

class Force_Num_Chan_Tensor_2d(object):
    '''
    Convert gray scale images to image with args.num_chan num channels.
    '''

    def __init__(self, args):
        super(Force_Num_Chan_Tensor_2d, self).__init__()

        def force_num_chan(tensor):
            existing_chan = tensor.size()[0]
            if not existing_chan == args.num_chan:
                return tensor.expand(args.num_chan, *tensor.size()[1:])
            return tensor

        self.transform = torchvision.transforms.Lambda(force_num_chan)

    def __call__(self, img, additional=None):
        return self.transform(img)

class ComposeTrans(object):
    '''
    composes multiple transformers
    '''
    def __init__(self, img_transformers, tnsr_transformers, args):
        super(ComposeTrans, self).__init__()
        self.img_transformers = img_transformers
        self.tnsr_transformers = tnsr_transformers
        self.args = args

    def __call__(self, img):
        transforms = self.transformers_sequence()
        composed_transform = torchvision.transforms.Compose(transforms)

        return composed_transform(img)

    def transformers_sequence(self):
        transforms = []

        # process image transformers
        for name, kwargs in self.parse_transformers(self.img_transformers):
           
            if name == 'colorjitter':
                transforms.append(
                    torchvision.transforms.ColorJitter(
                        brightness=kwargs['brightness'], 
                        contrast=kwargs['contrast'], 
                        saturation=kwargs['saturation'], 
                        hue=kwargs['hue']))

            if name == 'rand_horz_flip':
                transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))

            if name == 'rand_rotation':
                transforms.append(torchvision.transforms.RandomRotation((kwargs['min'], kwargs['max'])) )

            if name == 'rand_crop':
                 transforms.append(torchvision.transforms.RandomCrop((kwargs['h'], kwargs['w'])))

            if name == 'scale_2d':
                transforms.append(torchvision.transforms.Resize( (self.args.img_size[0], self.args.img_size[1]), interpolation=2))
        
        # process transformers programmed on tensors
        transforms.append(torchvision.transforms.ToTensor())

        for name, kwargs in self.parse_transformers(self.tnsr_transformers):
            if name == 'force_num_chan':
                transforms.append(Force_Num_Chan_Tensor_2d(self.args))
        
        if not self.args.computing_stats:
            mean = self.args.img_mean
            std = self.args.img_std
            transforms.append(torchvision.transforms.Normalize(mean, std, inplace=False))

        return transforms

    def parse_transformers(self, transformers):
        """
        Parse the list of transformers, given by configuration, into a list of
        tuple of the transformers name and a dictionary containing additional args.

        The transformer is assumed to be of the form 'name/arg1=value/arg2=value'

        :raw_transformers: list of strings [unparsed transformers]
        :returns: list of parsed transformers [list of (name,additional_args)]
        """
        raw_transformers = transformers

        transformers = []
        for t in raw_transformers:
            arguments = t.split('/')
            name = arguments[0]
            if name == '':
                raise Exception(EMPTY_NAME_ERR)

            kwargs = {}
            if len(arguments) > 1:
                for a in arguments[1:]:
                    splited = a.split('=')
                    var = splited[0]
                    val = splited[1] if len(splited) > 1 else None
                    if var == '':
                        raise Exception(EMPTY_NAME_ERR)

                    kwargs[var] = float(val)

            transformers.append((name, kwargs))

        return transformers
